from __future__ import print_function
from rpy2.robjects.packages import importr
from rpy2 import robjects

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import statsmodels.api as sm
import numpy as np, sys
import regreg.api as rr
from selection.randomized.api import randomization
from selection.adjusted_MLE.selective_MLE import M_estimator_map, solve_UMVU

def glmnet_sigma(X, y):
    robjects.r('''
                glmnet_cv = function(X,y){
                y = as.matrix(y)
                X = as.matrix(X)
                n = nrow(X)
                out = cv.glmnet(X, y, standardize=FALSE, intercept=FALSE)
                lam_1se = out$lambda.1se
                lam_min = out$lambda.min
                return(list(lam_min = n * as.numeric(lam_min), lam_1se = n* as.numeric(lam_1se)))
                }''')

    lambda_cv_R = robjects.globalenv['glmnet_cv']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)

    lam = lambda_cv_R(r_X, r_y)
    lam_min = np.array(lam.rx2('lam_min'))
    lam_1se = np.array(lam.rx2('lam_1se'))
    return lam_min, lam_1se


def sim_xy(n, p, nval, rho=0, s=5, beta_type=2, snr=1):
    robjects.r('''
    source('~/best-subset/bestsubset/R/sim.R')
    ''')

    r_simulate = robjects.globalenv['sim.xy']
    sim = r_simulate(n, p, nval, rho, s, beta_type, snr)
    X = np.array(sim.rx2('x'))
    y = np.array(sim.rx2('y'))
    X_val = np.array(sim.rx2('xval'))
    y_val = np.array(sim.rx2('yval'))
    Sigma = np.array(sim.rx2('Sigma'))
    beta = np.array(sim.rx2('beta'))
    sigma = np.array(sim.rx2('sigma'))

    return X, y, X_val, y_val, Sigma, beta, sigma

def tuned_lasso(X, y, X_val,y_val):
    robjects.r('''
        source('~/best-subset/bestsubset/R/lasso.R')
        tuned_lasso_estimator = function(X,Y,X.val,Y.val){
        Y = as.matrix(Y)
        X = as.matrix(X)
        Y.val = as.vector(Y.val)
        X.val = as.matrix(X.val)

        rel.LASSO = lasso(X,Y,intercept=FALSE, nrelax=10, nlam=50)
        LASSO = lasso(X,Y,intercept=FALSE,nlam=50)
        beta.hat.rellasso = as.matrix(coef(rel.LASSO))
        beta.hat.lasso = as.matrix(coef(LASSO))

        min.lam = min(rel.LASSO$lambda)
        max.lam = max(rel.LASSO$lambda)
        lam.seq = exp(seq(log(max.lam),log(min.lam),length=rel.LASSO$nlambda))

        muhat.val.rellasso = as.matrix(predict(rel.LASSO, X.val))
        muhat.val.lasso = as.matrix(predict(LASSO, X.val))

        err.val.rellasso = colMeans((muhat.val.rellasso - Y.val)^2)
        err.val.lasso = colMeans((muhat.val.lasso - Y.val)^2)

        opt_lam = ceiling(which.min(err.val.rellasso)/10)
        lambda.tuned = lam.seq[opt_lam]

        return(list(beta.hat.rellasso = beta.hat.rellasso[,which.min(err.val.rellasso)],
        beta.hat.lasso = beta.hat.lasso[,which.min(err.val.lasso)],
        lambda.tuned = lambda.tuned, lambda.seq = lam.seq))
        }''')

    r_lasso = robjects.globalenv['tuned_lasso_estimator']

    n, p = X.shape
    nval, _ = X_val.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_X_val = robjects.r.matrix(X_val, nrow=nval, ncol=p)
    r_y_val = robjects.r.matrix(y_val, nrow=nval, ncol=1)

    tuned_est = r_lasso(r_X, r_y, r_X_val, r_y_val)
    estimator_rellasso = np.array(tuned_est.rx2('beta.hat.rellasso'))
    estimator_lasso = np.array(tuned_est.rx2('beta.hat.lasso'))
    lam_tuned = np.array(tuned_est.rx2('lambda.tuned'))
    lam_seq = np.array(tuned_est.rx2('lambda.seq'))
    return estimator_rellasso, estimator_lasso, lam_tuned, lam_seq

def relative_risk(est, truth, Sigma):

    return (est-truth).T.dot(Sigma).dot(est-truth)/truth.T.dot(Sigma).dot(truth)

def inference_approx(n=500, p=100, nval=100, rho=0.35, s=5, beta_type=2, snr=0.2,
                     randomization_scale=np.sqrt(0.25)):

    while True:
        X, y, X_val, y_val, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
        true_mean = X.dot(beta)
        rel_LASSO, est_LASSO, lam_tuned, lam_seq = tuned_lasso(X, y, X_val, y_val)
        active_nonrand = (rel_LASSO != 0)
        nactive_nonrand = active_nonrand.sum()

        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n))

        X_val -= X_val.mean(0)[None, :]
        X_val /= (X_val.std(0)[None, :] * np.sqrt(nval))

        if p > n:
            sigma_est = np.std(y) / 2.
        else:
            ols_fit = sm.OLS(y, X).fit()
            sigma_est = np.linalg.norm(ols_fit.resid) / np.sqrt(n - p - 1.)

        loss = rr.glm.gaussian(X, y)
        epsilon = 1. / np.sqrt(n)


        lam_seq = np.linspace(0.75, 2.75, num= 100)\
                  *np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma_est
        err = np.zeros(100)
        for k in range(100):
            lam = lam_seq[k]
            W = np.ones(p) * lam
            penalty = rr.group_lasso(np.arange(p),
                                     weights=dict(zip(np.arange(p), W)), lagrange=1.)

            randomizer = randomization.isotropic_gaussian((p,), scale=randomization_scale)
            M_est = M_estimator_map(loss, epsilon, penalty, randomizer, randomization_scale=randomization_scale,
                                    sigma=sigma_est)

            M_est.solve_map()
            active = M_est._overall
            nactive = np.sum(active)
            Lasso_est = np.zeros(p)
            Lasso_est[active] = M_est.observed_opt_state[:nactive]
            err[k] = np.mean((y_val-X_val.dot(Lasso_est))**2.)

        lam = lam_seq[np.argmin(err)]
        randomizer = randomization.isotropic_gaussian((p,), scale=randomization_scale)
        W = np.ones(p) * lam
        penalty = rr.group_lasso(np.arange(p),
                                 weights=dict(zip(np.arange(p), W)), lagrange=1.)
        M_est = M_estimator_map(loss, epsilon, penalty, randomizer, randomization_scale=randomization_scale,
                                sigma=sigma_est)
        M_est.solve_map()
        active = M_est._overall
        nactive = np.sum(active)

        print("number of variables selected by randomized LASSO", nactive)
        print("number of variables selected by tuned LASSO", (rel_LASSO!=0).sum())
        true_signals = np.zeros(p, np.bool)
        true_signals[beta != 0] = 1
        screened_randomized = np.logical_and(active, true_signals).sum() /float(s)
        screened_nonrandomized = np.logical_and(active_nonrand, true_signals).sum() /float(s)
        false_positive_randomized = np.logical_and(active, ~true_signals).sum()/max(float(nactive), 1.)
        false_positive_nonrandomized = np.logical_and(active_nonrand, ~true_signals).sum()/max(float(nactive_nonrand),1.)

        true_set = np.asarray([u for u in range(p) if true_signals[u]])
        active_set = np.asarray([t for t in range(p) if active[t]])
        active_set_nonrand = np.asarray([q for q in range(p) if active_nonrand[q]])
        active_bool = np.zeros(nactive, np.bool)
        for x in range(nactive):
            active_bool[x] = (np.in1d(active_set[x],true_set).sum()>0)
        active_bool_nonrand= np.zeros(nactive_nonrand, np.bool)
        for y in range(nactive_nonrand):
            active_bool_nonrand[y] = (np.in1d(active_set_nonrand[y],true_set).sum()>0)

        true_target = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(true_mean)
        unad_sd = sigma_est * np.sqrt(np.diag(np.linalg.inv(X[:, active].T.dot(X[:, active]))))
        true_target_nonrand = np.linalg.inv(X[:, active_nonrand].T.dot(X[:, active_nonrand])).\
            dot(X[:, active_nonrand].T).dot(true_mean)
        unad_sd_nonrand = sigma_est * np.sqrt(np.diag(np.linalg.inv(X[:, active_nonrand].T.dot(X[:, active_nonrand]))))

        coverage_sel = 0.
        coverage_rand = 0.
        coverage_nonrand = 0.
        power_sel = 0.
        power_rand = 0.
        power_nonrand = 0.

        for k in range(nactive_nonrand):
            if (rel_LASSO[k]-(1.65 * unad_sd_nonrand[k])) <= true_target_nonrand[k] \
                    and (rel_LASSO[k]+(1.65 * unad_sd_nonrand[k])) >= true_target_nonrand[k]:
                coverage_nonrand += 1
            if active_bool_nonrand[k] == True and ((rel_LASSO[k]-(1.65 * unad_sd_nonrand[k])) > 0.
                                                   or (rel_LASSO[k]+(1.65 * unad_sd_nonrand[k])) <0.):
                power_nonrand += 1

        if nactive > 0:
            approx_MLE, var, mle_map, _, _, mle_transform = solve_UMVU(M_est.target_transform,
                                                                       M_est.opt_transform,
                                                                       M_est.target_observed,
                                                                       M_est.feasible_point,
                                                                       M_est.target_cov,
                                                                       M_est.randomizer_precision)

            mle_target_lin, mle_soln_lin, mle_offset = mle_transform
            approx_sd = np.sqrt(np.diag(var))

            if nactive == 1:
                approx_MLE = np.array([approx_MLE])
                approx_sd = np.array([approx_sd])

            for j in range(nactive):
                if (approx_MLE[j]-(1.65*approx_sd[j]))<= true_target[j] and (approx_MLE[j] + (1.65*approx_sd[j])) >= true_target[j]:
                    coverage_sel += 1
                if active_bool[j]==True and ((approx_MLE[j]-(1.65*approx_sd[j]))> 0. or (approx_MLE[j] + (1.65*approx_sd[j])) < 0.):
                    power_sel += 1
                if (M_est.target_observed[j]-(1.65*unad_sd[j]))<= true_target[j] and (M_est.target_observed[j]+(1.65*unad_sd[j])) >= true_target[j]:
                    coverage_rand += 1
                if active_bool[j]==True and ((M_est.target_observed[j]-(1.65*unad_sd[j]))>0. or (M_est.target_observed[j]+(1.65*unad_sd[j]))<0.):
                    power_rand += 1
            break

    target_par = beta

    ind_est = np.zeros(p)
    ind_est[active] = mle_target_lin.dot(M_est.target_observed) + \
                      mle_soln_lin.dot(M_est.observed_opt_state[:nactive]) + mle_offset
    ind_est /= np.sqrt(n)

    relaxed_Lasso = np.zeros(p)
    relaxed_Lasso[active] = M_est.target_observed / np.sqrt(n)

    Lasso_est = np.zeros(p)
    Lasso_est[active] = M_est.observed_opt_state[:nactive] / np.sqrt(n)

    selective_MLE = np.zeros(p)
    selective_MLE[active] = approx_MLE / np.sqrt(n)

    return (selective_MLE - target_par).sum() / float(nactive), \
           relative_risk(selective_MLE, target_par, Sigma), \
           relative_risk(relaxed_Lasso, target_par, Sigma), \
           relative_risk(ind_est, target_par, Sigma),\
           relative_risk(Lasso_est, target_par, Sigma),\
           relative_risk(rel_LASSO, target_par, Sigma),\
           relative_risk(est_LASSO, target_par, Sigma), \
           screened_randomized,\
           screened_nonrandomized,\
           false_positive_randomized, \
           false_positive_nonrandomized,\
           coverage_sel/max(float(nactive),1.),\
           coverage_rand/max(float(nactive),1.), \
           coverage_nonrand/max(float(nactive_nonrand),1.), \
           power_sel/float(s), \
           power_rand/float(s), \
           power_nonrand/float(s)

if __name__ == "__main__":

    ndraw = 150
    bias = 0.
    risk_selMLE = 0.
    risk_relLASSO = 0.
    risk_indest = 0.
    risk_LASSO = 0.
    risk_relLASSO_nonrand = 0.
    risk_LASSO_nonrand = 0.
    spower_rand = 0.
    spower_nonrand = 0.
    false_positive_randomized = 0.
    false_positive_nonrandomized = 0.
    coverage_sel = 0.
    coverage_rand = 0.
    coverage_nonrand = 0.
    power_sel = 0.
    power_rand = 0.
    power_nonrand = 0.

    for i in range(ndraw):
        approx = inference_approx(n=500, p=100, nval=100, rho=0.35, s=5, beta_type=2, snr=0.10)
        if approx is not None:
            bias += approx[0]
            risk_selMLE += approx[1]
            risk_relLASSO += approx[2]
            risk_indest += approx[3]
            risk_LASSO += approx[4]
            risk_relLASSO_nonrand += approx[5]
            risk_LASSO_nonrand += approx[6]
            spower_rand += approx[7]
            spower_nonrand += approx[8]
            false_positive_randomized += approx[9]
            false_positive_nonrandomized += approx[10]
            coverage_sel += approx[11]
            coverage_rand += approx[12]
            coverage_nonrand += approx[13]
            power_sel += approx[14]
            power_rand += approx[15]
            power_nonrand += approx[16]

        sys.stderr.write("overall_bias" + str(bias / float(i + 1)) + "\n")
        sys.stderr.write("overall_selrisk" + str(risk_selMLE / float(i + 1)) + "\n")
        sys.stderr.write("overall_relLASSOrisk" + str(risk_relLASSO / float(i + 1)) + "\n")
        sys.stderr.write("overall_indepestrisk" + str(risk_indest / float(i + 1)) + "\n")
        sys.stderr.write("overall_LASSOrisk" + str(risk_LASSO / float(i + 1)) + "\n")
        sys.stderr.write("overall_relLASSOrisk_norand" + str(risk_relLASSO_nonrand / float(i + 1)) + "\n")
        sys.stderr.write("overall_LASSOrisk_norand" + str(risk_LASSO_nonrand / float(i + 1)) + "\n")

        sys.stderr.write("overall_LASSO_rand_spower" + str(spower_rand / float(i + 1)) + "\n")
        sys.stderr.write("overall_LASSO_norand_spower" + str(spower_nonrand / float(i + 1)) + "\n")
        sys.stderr.write("overall_LASSO_rand_falsepositives" + str(false_positive_randomized / float(i + 1)) + "\n")
        sys.stderr.write("overall_LASSO_norand_falsepositives" + str(false_positive_nonrandomized / float(i + 1)) + "\n")

        sys.stderr.write("selective coverage" + str(coverage_sel / float(i + 1)) + "\n")
        sys.stderr.write("randomized coverage" + str(coverage_rand / float(i + 1)) + "\n")
        sys.stderr.write("nonrandomized coverage" + str(coverage_nonrand / float(i + 1)) + "\n")

        sys.stderr.write("selective power" + str(power_sel / float(i + 1)) + "\n")
        sys.stderr.write("randomized power" + str(power_rand / float(i + 1)) + "\n")
        sys.stderr.write("nonrandomized power" + str(power_nonrand / float(i + 1)) + "\n")

        sys.stderr.write("iteration completed" + str(i) + "\n")








