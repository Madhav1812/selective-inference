import numpy as np
from scipy.optimize import minimize, bisect
from selection.sampling.randomized.tests.test_lasso_fixedX_saturated import test_lasso
from selection.sampling.randomized.tests.test_lasso_fixedX_saturated import selection
import selection.sampling.randomized.api as randomized
from scipy.stats import laplace, probplot, uniform
from selection.algorithms.lasso import instance
import regreg.api as rr
from matplotlib import pyplot as plt

plt.figure()
plt.ion()


def joint_Gaussian_parameters(X, y, active, signs, j, epsilon, lam, sigma, tau):
    """ Sigma_inv_mu computed for beta_{E,j}^*=0
    """
    n, p = X.shape
    nactive = np.sum(active)
    ninactive = p-nactive

    mat = np.linalg.pinv(X[:, active])
    eta = mat[j, :]
    eta_norm_sq = np.linalg.norm(eta) **2

    #from Snigdha's R code:
    #XE = X[:,active]
    #if nactive>1:
    #    keep = np.ones(nactive, dtype=bool)
    #    keep[j] = False
    #    eta = (np.identity(n)- np.dot(XE[:, keep], np.linalg.pinv(XE[:,keep]))).dot(XE[:,j])
    #else:
    #    eta = np.true_divide(XE[:,j], np.linalg.norm(XE[:,j])**2)


    c = np.true_divide(eta, eta_norm_sq)

    A = np.zeros((p, p + 1))
    A[:, 0] = -np.dot(X.T, c)
    A[:, 1:(nactive + 1)] = np.dot(X.T, X[:, active])
    A[:nactive, 1:(nactive + 1)] += epsilon * np.identity(nactive)
    A[nactive:, (nactive + 1):] = lam * np.identity(ninactive)

    fixed_part = np.dot(np.identity(n) - np.outer(c, eta), y)

    gamma = -np.dot(X.T, fixed_part)
    gamma[:nactive] += lam * signs

    v = np.zeros(p + 1)
    v[0] = 1

    Sigma_inv = np.true_divide(np.dot(A.T, A), tau ** 2) + np.true_divide(np.outer(v, v), eta_norm_sq*(sigma ** 2))
    Sigma_inv_mu = np.true_divide(np.dot(A.T, gamma), tau ** 2)

    return Sigma_inv, Sigma_inv_mu



def log_selection_probability(param, Sigma_full, Sigma_inv, Sigma_inv_mu, sigma,
                              nactive, ninactive, signs, betaE, eta_norm_sq):
    #print 'param value', param
    p = nactive+ninactive
    Sigma_inv_mu_modified = Sigma_inv_mu.copy()
    Sigma_inv_mu_modified[0] += param/(eta_norm_sq*(sigma ** 2))

    initial_guess = np.zeros(p + 1)
    initial_guess[1:(nactive + 1)] = betaE
    initial_guess[(nactive+1):] = np.random.uniform(-1,1, ninactive)

    bounds = ((None, None),)
    for i in range(nactive):
        if signs[i] < 0:
            bounds += ((None, 0),)
        else:
            bounds += ((0, None),)
    bounds += ((-1, 1),) * ninactive

    def chernoff(x):
        return np.inner(x, Sigma_inv.dot(x))/2 - np.inner(Sigma_inv_mu_modified, x)

    def barrier(x):
        # Ax\leq b
        A = np.zeros((nactive+2*ninactive, 1+nactive+ninactive))
        A[:nactive, 1:(nactive+1)] = -np.diag(signs)
        A[nactive:(nactive+ninactive), (nactive+1):] = np.identity(ninactive)
        A[(nactive+ninactive):, (nactive+1):] = -np.identity(ninactive)
        b = np.zeros(nactive+2*ninactive)
        b[nactive:] = 1

        if all(b-np.dot(A,x)>=np.power(10,-9)):
            return np.sum(np.log(1+np.true_divide(1,b-np.dot(A,x))))

        return b.shape[0]*np.log(1+10**9)

    def objective(x):
        return chernoff(x)+barrier(x)

    res = minimize(objective, x0=initial_guess) # , bounds=bounds)
    #print nactive, ninactive
    #print signs
    #print nactive
    #print res.x
    mu = np.dot(Sigma_full, Sigma_inv_mu_modified)
    return - np.true_divide(np.inner(mu, Sigma_inv_mu_modified), 2) - res.fun
    #return -np.inner(mu, Sigma_inv_mu_modified) / 2 - objective(res.x)


def compute_mle(observed_vector, Sigma_full, Sigma_inv, Sigma_inv_mu, sigma,
                nactive, ninactive, signs, eta_norm_sq):

    betaE = observed_vector[1:(1+nactive)]

    def objective_mle(param):
        Sigma_inv_mu_modified = Sigma_inv_mu.copy()
        Sigma_inv_mu_modified[0] += param / (eta_norm_sq*(sigma ** 2))
        mu = np.dot(Sigma_full, Sigma_inv_mu_modified)
        return -np.inner(observed_vector, Sigma_inv_mu_modified)+\
               np.true_divide(np.inner(mu,Sigma_inv_mu_modified), 2) + \
               log_selection_probability(param, Sigma_full, Sigma_inv,Sigma_inv_mu, sigma, nactive, ninactive, signs, betaE, eta_norm_sq)

    initial_guess_mle = 0
    res_mle = minimize(objective_mle, x0=initial_guess_mle)

    return res_mle.x



def intervals(n=50, p=10, s=0, alpha=0.1):

    X, y, true_beta, nonzero, sigma = instance(n=n, p=p, random_signs=True, s=s, snr =2, sigma=1., rho=0)
    #print sigma
    #print true_beta
    random_Z = np.random.standard_normal(p)

    lam, epsilon, active, betaE, cube = selection(X,y, random_Z)
    if lam < 0:
        print "no active covariates"
        return -1, -1

    nactive = np.sum(active)
    #print 'size of the active set', nactive
    #print 'active set', active

    tau = 1.
    inactive = ~active
    signs = np.sign(betaE)
    ninactive = np.sum(inactive)
    active_set = np.where(active)[0]

    coverage = 0

    observed_vector = np.zeros(p+1)
    observed_vector[1:(nactive+1)] = betaE
    observed_vector[(1+nactive):] = cube

    Sigma_inv = [np.array((p+1,p+1)) for i in range(nactive)]
    Sigma_full = [np.array((p+1, p+1)) for i in range(nactive)]
    Sigma_inv_mu = [np.zeros(p+1) for i in range(nactive)]

    beta_mle = np.zeros(nactive)

    if set(nonzero).issubset(active_set):
        for j, idx in enumerate(active_set):
            Sigma_inv[j], Sigma_inv_mu[j] = joint_Gaussian_parameters(X,y, active, signs, j, epsilon, lam, sigma, tau)
            Sigma_full[j] = np.linalg.inv(Sigma_inv[j])

            eta = np.linalg.pinv(X[:, active])[j, :]
            eta_norm_sq = np.linalg.norm(eta)**2
            observed_vector[0] = np.inner(eta, y)

            #beta_mle[j] = compute_mle(observed_vector.copy(), Sigma_full[j], Sigma_inv[j], Sigma_inv_mu[j],
            #                          sigma, nactive, ninactive, signs, eta_norm_sq)


    print "MLE", beta_mle

    beta_mle = np.zeros(nactive)

    _, _, all_observed, all_variances, all_samples = test_lasso(X, y, nonzero, sigma, lam, epsilon, active, betaE,
                                                                cube, random_Z,
                                                                beta_reference=beta_mle,
                                                                randomization_distribution="normal")



    if set(nonzero).issubset(active_set):
        for j, idx in enumerate(active_set):
            truth = true_beta[idx]

            eta = np.linalg.pinv(X[:, active])[j, :]
            eta_norm_sq = np.linalg.norm(eta)**2
            observed_vector[0] = np.inner(eta, y)

            param_values = np.linspace(-10, 10, num=400)
            log_sel_prob_param = np.zeros(param_values.shape[0])

            #for i in range(param_values.shape[0]):
            #     log_sel_prob_param[i] = log_selection_probability(param_values[i], Sigma_full, Sigma_inv, Sigma_inv_mu, sigma,
            #                                                     nactive, ninactive, signs, betaE, eta_norm_sq)
            plt.clf()
            #plt.title("Log of selection probabilities")
            #plt.plot(param_values, log_sel_prob_param)
            #plt.pause(0.01)

            obs = np.inner(eta, y) # same as np.inner(eta, y)
            sd = np.linalg.norm(eta)*sigma

            indicator = np.array(all_samples[j,:]<all_observed[j], dtype =int)
            #indicator = np.array(np.abs(all_samples[j, :]) > all_observed[j], dtype=int)

            pop = all_samples[j,:]
            variance = all_variances[j]
            #print "variance", variance
            #print (np.linalg.norm(eta)**2)*sigma
            log_sel_prob_ref = log_selection_probability(beta_mle[j].copy(), Sigma_full[j].copy(), Sigma_inv[j].copy(), Sigma_inv_mu[j].copy(),
                                                         sigma,
                                                         nactive, ninactive, signs, betaE, eta_norm_sq)

            def pvalue_by_tilting(param_value, variance=variance, pop=pop, indicator=indicator,
                                  ref_param = beta_mle[j]):
                 log_sel_prob_param = log_selection_probability(param_value, Sigma_full[j], Sigma_inv[j], Sigma_inv_mu[j],
                                                                sigma,
                                                                nactive, ninactive, signs, betaE, eta_norm_sq)
                 log_LR = pop*(param_value-ref_param)/(2*variance)-(param_value**2-ref_param**2)/(2*variance)
                 log_LR += log_sel_prob_ref - log_sel_prob_param
                 return np.clip(np.sum(np.multiply(indicator, np.exp(log_LR)))/ indicator.shape[0], 0,1)

            # #print 'pvalue at the truth', pvalue_by_tilting(0)
            # #print 'pvalue at the truth', pvalue_by_tilting(0)
            #
            # #print 'param value', param_values

            pvalues = [pvalue_by_tilting(param_values[i]) for i in range(param_values.shape[0])]
            pvalues = np.asarray(pvalues, dtype=np.float32)
            #print pvalues
            plt.title("Tilted p-values")
            plt.plot(param_values, pvalues)
            plt.pause(0.01)

            #accepted_indices = np.multiply(np.array(pvalues>alpha/2), np.array(pvalues<1.-alpha/2))
            accepted_indices = np.array(pvalues > alpha)

            if np.sum(accepted_indices)==0:
                 L=0
                 U=0
            else:
                 L = np.min(param_values[accepted_indices])
                 U = np.max(param_values[accepted_indices])

            #L = param_values[np.argmin(np.abs(pvalues-(alpha/2)))]
            #U = param_values[np.argmin(np.abs(pvalues-1.+(alpha/2)))]
            #print "truth", truth
            if (L<truth) and (U> truth):
                 coverage +=1
            if (U < truth) and (L > truth):
                 coverage += 1
            print "interval", L, U

    return coverage, nactive

total_coverage = 0
total_number = 0


for i in range(30):
    print "\n"
    print "iteration", i
    coverage, nactive = intervals()
    if coverage>=0:
        total_coverage += coverage
        total_number += nactive
print "number covered out of", total_coverage, total_number
print "total coverage", np.true_divide(total_coverage, total_number)


while True:
    plt.pause(0.05)
plt.show()










