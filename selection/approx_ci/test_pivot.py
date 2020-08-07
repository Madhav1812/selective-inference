from __future__ import division, print_function

import numpy as np
from selection.randomized.lasso import lasso, carved_lasso, selected_targets, full_targets, debiased_targets
from selection.tests.instance import gaussian_instance, exp_instance, normexp_instance, mixednormal_instance, laplace_instance
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import matplotlib.pyplot as plt
import scipy.stats as sp
from scipy.sparse import vstack
import seaborn as sns
import pandas as pd

from statsmodels.distributions.empirical_distribution import ECDF
from approx_reference import approx_reference, approx_reference_adaptive, approx_density, approx_ci

np.random.seed(1812)

def test_approx_pivot(n= 500,
                      p= 100,
                      signal_fac= 1.,
                      s= 5,
                      sigma= 1.,
                      rho= 0.40,
                      randomizer_scale= 1.):

    inst = gaussian_instance
    signal = np.sqrt(signal_fac * 2. * np.log(p))

    while True:
        X, y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        n, p = X.shape
        sel = np.append(np.repeat(True, n / 2), np.repeat(False, n - n / 2))
        np.random.shuffle(sel)
        mod = np.logical_not(sel)
        X_sel = X[sel]
        y_sel = y[sel]
        X_mod = X[mod]
        y_mod = y[mod]
        if n>p:
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
            dispersion_sel = np.linalg.norm(y_sel - X_sel.dot(np.linalg.pinv(X_sel).dot(y_sel))) ** 2/ (n/2 - p)
            sigma_ = np.sqrt(dispersion)
            sigma_sel = np.sqrt(dispersion_sel)
        else:
            dispersion = None
            dispersion_sel = None
            sigma_ = np.std(y)
            sigma_sel = np.std(y_sel)

        print("sigma estimated and true ", sigma, sigma_, sigma_sel)

        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_
        W_sel = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_sel
        l_lasso = np.sqrt(2*np.log(p))*sigma_
        conv = lasso.gaussian(X,
                              y,
                              W,
                              randomizer_scale=randomizer_scale * sigma_)
        conv_sel = lasso.gaussian(X_sel,
                              y_sel,
                              W_sel,
                              randomizer_scale=randomizer_scale * sigma_sel)

        signs = conv.fit()
        signs_sel = conv_sel.fit()
        nonzero = signs != 0
        nonzero_sel = signs_sel != 0
        select_signs = signs[nonzero]
        beta_split = np.linalg.inv(X_mod.T.dot(X_mod)).dot(X_mod.T.dot(y_mod))
        cov_split = np.linalg.inv(X_mod.T.dot(X_mod)) * sigma_sel
        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=dispersion)
        (observed_target_split,
         cov_target_split,
         cov_target_score_split,
         alternatives_split) = selected_targets(conv_sel.loglike,
                                          conv_sel._W,
                                          nonzero_sel,
                                          dispersion=dispersion_sel)
        grid_num = 501
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        X_S = np.asarray(X[:,nonzero])
        X_U = np.asarray(X[:,np.logical_not(nonzero)])
        X_M = np.linalg.inv(np.transpose(X_S).dot(X_S)).dot(np.transpose(X_S))
        P_S = X_S.dot(X_M)
        NP_S = np.identity(np.shape(P_S)[0]) - P_S
        A_0 = np.vstack((X_U.T.dot(NP_S)/l_lasso,-X_U.T.dot(NP_S)/l_lasso))
        b_0 = np.hstack((np.ones(X_U.shape[1])-X_U.T.dot(X_M.T).dot(select_signs),np.ones(X_U.shape[1])+X_U.T.dot(X_M.T).dot(select_signs)))
        A_1 = -np.diag(select_signs).dot(np.linalg.inv(X_S.T.dot(X_S))).dot(X_M)
        b_1 = -l_lasso*np.diag(select_signs).dot(np.linalg.inv(X_S.T.dot(X_S))).dot(select_signs)
        A = np.vstack((A_0,A_1))
        b = np.empty(b_0.shape[0]+b_1.shape[0])
        b[0:b_0.shape[0]] = b_0
        b[b_0.shape[0]:b_0.shape[0]+b_1.shape[0]] = b_1
        pivot = []
        naive_pivot = []
        lee_pivot = []
        split_pivot = []
        ci_lower = np.repeat(-float('inf'), nonzero.sum())
        ci_upper = np.repeat(float('inf'), nonzero.sum())
        length = np.repeat(float('inf'), nonzero.sum())
        coverage = np.repeat(False, nonzero.sum())
        ci_lower_naive = np.repeat(-float('inf'), nonzero.sum())
        ci_upper_naive = np.repeat(float('inf'), nonzero.sum())
        length_naive = np.repeat(float('inf'), nonzero.sum())
        coverage_naive = np.repeat(False, nonzero.sum())
        ci_lower_lee = np.repeat(-float('inf'), nonzero.sum())
        ci_upper_lee = np.repeat(float('inf'), nonzero.sum())
        length_lee = np.repeat(float('inf'), nonzero.sum())
        coverage_lee = np.repeat(False, nonzero.sum())
        ci_lower_split = np.repeat(-float('inf'), nonzero.sum())
        ci_upper_split = np.repeat(float('inf'), nonzero.sum())
        length_split = np.repeat(float('inf'), nonzero.sum())
        coverage_split = np.repeat(False, nonzero.sum())
        for m in range(nonzero.sum()):
            observed_target_uni = (observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(cov_target)[m]).reshape((1,1))
            cov_target_score_uni = cov_target_score[m,:].reshape((1, p))
            mean_parameter = beta_target[m]
            grid = np.linspace(- 25., 25., num=grid_num)
            grid_indx_obs = np.argmin(np.abs(grid - observed_target_uni))
            e_vector = np.zeros(nonzero.sum())
            e_vector[m] = 1
            eta = X_M.T.dot(e_vector)
            z = (np.identity(np.shape(P_S)[0])-(eta.dot(eta.T)/(eta.T.dot(eta)))).dot(np.asarray(y))
            pos_A = A.dot(eta/eta.T.dot(eta)) > 0
            neg_A = A.dot(eta/eta.T.dot(eta)) < 0
            zer_A = A.dot(eta/eta.T.dot(eta)) == 0
            nonz_A = np.logical_not(zer_A)
            lim_b = b - A.dot(z)
            if np.sum(zer_A) == 0:
                rat_b = np.divide(lim_b,A.dot(eta/eta.T.dot(eta)))
                delta_low = np.max(rat_b[neg_A])
                delta_high = np.min(rat_b[pos_A])
            else:
                delta_0 = np.min(lim_b[zer_A])
                rat_b = np.empty(np.shape(lim_b)[0])
                rat_b[nonz_A] = np.divide(lim_b[nonz_A],A.dot(eta / eta.T.dot(eta))[nonz_A])
                delta_low = np.max(rat_b[neg_A])
                delta_high = np.min(rat_b[pos_A])
            approx_log_ref= approx_reference(grid,
                                             observed_target_uni,
                                             cov_target_uni,
                                             cov_target_score_uni,
                                             conv.observed_opt_state,
                                             conv.cond_mean,
                                             conv.cond_cov,
                                             conv.logdens_linear,
                                             conv.A_scaling,
                                             conv.b_scaling)

            area_cum, area_cum_naive, area_cum_lee = approx_density(grid,
                                                                    mean_parameter,
                                                                    cov_target_uni,
                                                                    approx_log_ref,
                                                                    delta_low,
                                                                    delta_high)

            param_grid = np.linspace(-100, 100, num=2001)
            ci_lower[m], ci_upper[m] = approx_ci(param_grid,
                                           grid,
                                           cov_target_uni,
                                           approx_log_ref,
                                           grid_indx_obs,
                                           delta_low,
                                           delta_high)

            if ci_lower[m] > -float('inf'):
                if ci_upper[m] < float('inf'):
                    length[m] = ci_upper[m] - ci_lower[m]


            if ci_lower[m] < 0 < ci_upper[m]:
                coverage[m]=True

            pivot.append(min(2*(1. - area_cum[grid_indx_obs]), 2*(area_cum[grid_indx_obs])))
            naive_pivot.append(min(2*(1 - area_cum_naive[grid_indx_obs]),2*(area_cum_naive[grid_indx_obs])))
            lee_pivot.append(min(2*(1. - area_cum[grid_indx_obs]),2*area_cum[grid_indx_obs]))
            print("variable completed ", m+1)
        for m in range(nonzero_sel.sum()):
            observed_target_uni = (observed_target_split[m]).reshape((1,))
            cov_target_uni = (np.diag(cov_target_split)[m]).reshape((1,1))
            cov_target_score_uni = cov_target_score_split[m,:].reshape((1, p))
            mean_parameter = beta_split[m]
            grid = np.linspace(- 25., 25., num=grid_num)
            grid_indx_obs = np.argmin(np.abs(grid - observed_target_uni))
            approx_log_ref = approx_reference(grid,
                                             observed_target_uni,
                                             cov_target_uni,
                                             cov_target_score_uni,
                                             conv.observed_opt_state,
                                             conv.cond_mean,
                                             conv.cond_cov,
                                             conv.logdens_linear,
                                             conv.A_scaling,
                                             conv.b_scaling)

            _, area_cum_split, _ = approx_density(grid,
                                            mean_parameter,
                                            cov_target_uni,
                                            approx_log_ref
                                            )

            param_grid = np.linspace(-100, 100, num=2001)

            ci_lower[m], ci_upper[m]= approx_ci(param_grid,
                                           grid,
                                           cov_target_uni,
                                           approx_log_ref,
                                           grid_indx_obs
                                                )

            if ci_lower[m] > -float('inf'):
                if ci_upper[m] < float('inf'):
                    length[m] = ci_upper[m] - ci_lower[m]


            if ci_lower[m] <= 0 <= ci_upper[m]:
                coverage[m]=True

            split_pivot.append(min(2*(1. - area_cum_split[grid_indx_obs]),area_cum_split[grid_indx_obs]))
            print("variable completed ", m+1)
        total_length = np.sum(length)
        total_coverage = np.sum(coverage)
        print(coverage)
        print(ci_lower)
        print(ci_upper)
        return pivot, naive_pivot, lee_pivot, split_pivot, total_length, total_coverage, nonzero.sum()


def test_approx_pivot_adaptive(n=200,
                               p=50,
                               signal_fac=1.2,
                               s=5,
                               sigma=1.,
                               rho=0.40,
                               randomizer_scale=1.):
    inst = gaussian_instance
    signal = np.sqrt(signal_fac * 2. * np.log(p))

    while True:
        X, y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        n, p = X.shape
        sel = np.append(np.repeat(True, n / 2), np.repeat(False, n - n / 2))
        np.random.shuffle(sel)
        mod = np.logical_not(sel)
        X_sel = X[:sel]
        y_sel = y[:sel]
        X_mod = X[:mod]
        y_mod = y[:mod]
        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        if n > p:
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
            dispersion_sel = np.linalg.norm(y_sel - X_sel.dot(np.linalg.pinv(X_sel).dot(y_sel))) ** 2 / (n/2 - p)
            sigma_ = np.sqrt(dispersion)
            sigma_sel = np.sqrt(dispersion_sel)
        else:
            dispersion = None
            sigma_ = np.std(y)

        scaling = np.linalg.inv(X.T.dot(X))/(2.* sigma_)
        scaling_sel = np.linalg.inv(X_sel.T.dot(X_sel))/(2.*sigma_sel)
        W = 1./np.abs(scaling.dot(X.T.dot(y)))
        W_sel = 1. / np.abs(scaling.dot(X_sel.T.dot(y_sel)))
        W_split = 1./np.abs(scaling_sel.dot(X_sel.T.dot(y_sel)))
        conv = lasso.gaussian(X,
                              y,
                              W,
                              randomizer_scale=randomizer_scale * sigma_)
        conv_split = lasso.gaussian(X_sel,
                              y_sel,
                              W_sel,
                              randomizer_scale=randomizer_scale * sigma_sel)
        signs = conv.fit()
        signs_sel = conv_split.fit()
        nonzero = signs != 0
        nonzero_sel = signs_sel != 0

        beta_split = np.linalg.inv(X_mod.T.dot(X_mod)).dot(X_mod.T.dot(y_mod))
        cov_split = np.linalg.inv(X_mod.T.dot(X_mod)) * sigma_sel

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=dispersion)

        grid_num = 501
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        pivot = []
        naive_pivot = []
        split_pivot = []
        ci_lower = np.repeat(-float('inf'), nonzero.sum())
        ci_upper = np.repeat(float('inf'), nonzero.sum())
        length = np.repeat(float('inf'), nonzero.sum())
        coverage = np.repeat(False, nonzero.sum())
        for m in range(nonzero.sum()):
            observed_target_uni = (observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(cov_target)[m]).reshape((1, 1))
            cov_target_score_uni = cov_target_score[m, :].reshape((1, p))
            mean_parameter = beta_target[m]
            mean_split_para = beta_split[m]
            cov_target_split = cov_split[m, m]
            grid = np.linspace(- 20., 20., num=grid_num)
            grid_indx_obs = np.argmin(np.abs(grid - observed_target_uni))

            approx_log_ref = approx_reference_adaptive(grid,
                                                       observed_target_uni,
                                                       cov_target_uni,
                                                       cov_target_score_uni,
                                                       conv.observed_opt_state,
                                                       conv.cond_mean,
                                                       conv.cond_cov,
                                                       conv.logdens_linear,
                                                       conv.A_scaling,
                                                       conv.b_scaling,
                                                       conv.initial_subgrad,
                                                       conv.feature_weights,
                                                       conv.observed_score_state,
                                                       nonzero,
                                                       scaling)

            area_cum, area_cum_naive, _ = approx_density(grid,
                                      mean_parameter,
                                      cov_target_uni,
                                      approx_log_ref,
                                      -float('inf'),
                                      float('inf'))
            area_cum_split = approx_density(grid,
                                                      mean_split_para,
                                                      cov_target_split)
            param_grid = np.linspace(-2 * mean_parameter, 2 * mean_parameter, num=grid_num)
            ci_lower[m], ci_upper[m] = approx_ci(param_grid,
                                                 grid,
                                                 cov_split,
                                                 approx_log_ref,
                                                 grid_indx_obs,
                                                 -float('inf'),
                                                 float('inf'))

            if ci_lower[m] > -float('inf'):
                if ci_upper[m] < float('inf'):
                    length[m] = ci_upper[m] - ci_lower[m]

            if ci_lower[m] <= 0 <= ci_upper[m]:
                coverage[m] = True
            pivot.append(min(2*(1 -area_cum[grid_indx_obs]),2*(area_cum[grid_indx_obs])))

            naive_pivot.append(min(2*(1 -area_cum_naive[grid_indx_obs]),2*(area_cum_naive[grid_indx_obs])))
            split_pivot.append(min(2*(1 -area_cum_split[grid_indx_obs]),2*(area_cum_split[grid_indx_obs])))
            print("variable completed ", m + 1)
        total_length = np.sum(length)
        total_coverage = np.sum(coverage)
        print(coverage)
        print(ci_lower)
        print(ci_upper)
        return pivot, naive_pivot, split_pivot, total_length, total_coverage


def test_approx_pivot_carved(n= 100,
                             p= 50,
                             signal_fac= 1.,
                             s= 5,
                             sigma= 1.,
                             rho= 0.40,
                             split_proportion=0.50):

    inst = laplace_instance
    signal = np.sqrt(signal_fac * 2. * np.log(p))

    while True:
        X, y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))
        n, p = X.shape
        if n>p:
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
            sigma_ = np.sqrt(dispersion)
        else:
            dispersion = None
            sigma_ = np.std(y)

        print("sigma estimated and true ", sigma, sigma_)
        randomization_cov = ((sigma_ ** 2) * ((1. - split_proportion) / split_proportion)) * sigmaX
        lam_theory = sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

        conv = carved_lasso.gaussian(X,
                                     y,
                                     noise_variance=sigma_ ** 2.,
                                     rand_covariance="True",
                                     randomization_cov=randomization_cov / float(n),
                                     feature_weights=np.ones(X.shape[1]) * lam_theory,
                                     subsample_frac=split_proportion)

        signs = conv.fit()
        nonzero = signs != 0

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=dispersion)

        grid_num = 501
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        pivot = []
        for m in range(nonzero.sum()):
            observed_target_uni = (observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(cov_target)[m]).reshape((1, 1))
            cov_target_score_uni = cov_target_score[m, :].reshape((1, p))
            mean_parameter = beta_target[m]
            grid = np.linspace(- 25., 25., num=grid_num)
            grid_indx_obs = np.argmin(np.abs(grid - observed_target_uni))

            approx_log_ref = approx_reference(grid,
                                              observed_target_uni,
                                              cov_target_uni,
                                              cov_target_score_uni,
                                              conv.observed_opt_state,
                                              conv.cond_mean,
                                              conv.cond_cov,
                                              conv.logdens_linear,
                                              conv.A_scaling,
                                              conv.b_scaling,
                                              )

            area_cum = approx_density(grid,
                                      mean_parameter,
                                      cov_target_uni,
                                      approx_log_ref)

            pivot.append(1. - area_cum[grid_indx_obs])
            print("variable completed ", m + 1)
        return pivot

def ECDF_pivot(nsim=300):
    _pivot = []
    _pivot_naive = []
    _pivot_lee = []
    _pivot_split = []
    average_length = np.repeat(float('inf'),nsim)
    coverage = np.repeat(1, nsim)
    for i in range(nsim):
        test_pivot, test_pivot_naive, test_pivot_lee, test_pivot_split, average_length[i], coverage[i], varno = test_approx_pivot(n= 200,
                                                                         p= 50,
                                                                         signal_fac= 0.25,
                                                                         s= 5,
                                                                         sigma= 1.,
                                                                         rho= 0.40,
                                                                         randomizer_scale= 1.)
        _pivot.extend(test_pivot)
        _pivot_naive.extend(test_pivot_naive)
        _pivot_lee.extend(test_pivot_lee)
        _pivot_split.extend(test_pivot_split)
        print("iteration completed ", i)
    plt.clf()
    ecdf_MLE = ECDF(np.asarray(_pivot))
    ecdf_MLE_naive = ECDF(np.asarray(_pivot_naive))
    ecdf_MLE_lee = ECDF(np.asarray(_pivot_lee))
    ecdf_MLE_split = ECDF(np.asarray(_pivot_split))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    plt.plot(grid, ecdf_MLE_naive(grid), c='red', marker='o')
    plt.plot(grid, ecdf_MLE_lee(grid), c='green', marker='^')
    plt.plot(grid, ecdf_MLE_split(grid), c='yellow', marker='o')
    plt.plot(grid, grid, 'k--')
    plt.show()

#ECDF_pivot(nsim=50)

def CI_pivot(nsim=300):
    _pivot = []
    _pivot_naive = []
    _pivot_lee = []
    _pivot_split = []
    total = 0
    total_length = np.repeat(float('inf'), nsim)
    total_coverage = np.repeat(1, nsim)
    for i in range(nsim):
        test_pivot, test_pivot_naive, test_pivot_lee, test_pivot_split, total_length[i], total_coverage[i], varno = test_approx_pivot(n=200,
                                                                                           p=50,
                                                                                           signal_fac=0.25,
                                                                                           s=5,
                                                                                           sigma=1.,
                                                                                           rho=0.40,
                                                                                           randomizer_scale=1.)
        _pivot.extend(test_pivot)
        _pivot_naive.extend(test_pivot_naive)
        _pivot_lee.extend(test_pivot_lee)
        _pivot_split.extend(test_pivot_split)
        print("iteration completed ", i)
        total = total + varno
    average = np.sum(total_length)/total
    cover = np.sum(total_coverage)/total
    print(average)
    print(cover)

CI_pivot(nsim=5)

def ECDF_pivot_adapt(nsim=300):
    _pivot = []
    _pivot_split = []
    for i in range(nsim):
        test_pivot, _ , test_pivot_split, _, _ = test_approx_pivot_adaptive(n=1000,
                                          p=200,
                                          signal_fac=0.25,
                                          s=15,
                                          sigma=1.,
                                          rho=0.40,
                                          randomizer_scale=1.)
        _pivot.extend(test_pivot)
        _pivot_split.extend(test_pivot_split)
        print("iteration completed ", i)
    plt.clf()
    ecdf_MLE = ECDF(np.asarray(_pivot))
    print(np.asarray(_pivot))
    ecdf_MLE_split = ECDF(np.asarray(_pivot_split))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    plt.plot(grid, ecdf_MLE_split(grid), c='red', marker='o')
    plt.plot(grid, grid, 'k--')
    plt.show()

#ECDF_pivot_adapt(nsim=10)
def ECDF_pivot_naive(nsim=300):
    _pivot = []
    _pivot_adaptive = []
    for i in range(nsim):
        _, test_adaptive_naive, _ = test_approx_pivot_adaptive(n=200,
                                          p=50,
                                          signal_fac=0.25,
                                          s=5,
                                          sigma=1.,
                                          rho=0.40,
                                          randomizer_scale=1.)
        _pivot_adaptive.extend(test_adaptive_naive)
        _, test_naive, _ , _, _, _, _= test_approx_pivot(n=200,
                                          p=50,
                                          s=5,
                                          sigma=1.,
                                          signal_fac=0.25,
                                          rho=0.40,
                                          randomizer_scale=1.)
        _pivot.extend(test_naive)
        print("iteration completed ", i)
    ecdf_MLE = ECDF(np.asarray(_pivot))
    ecdf_MLE_adaptive = ECDF(np.asarray(_pivot_adaptive))
    grid = np.linspace(0, 1, 101)
    ecdf = np.append(ecdf_MLE(grid), ecdf_MLE_adaptive(grid))
    method_1 = np.repeat("LASSO", np.shape(grid)[0])
    method_2 = np.repeat("Adaptive LASSO",np.shape(grid)[0])
    grid = np.append(grid,grid)
    method = np.append(method_1,method_2)
    data=pd.DataFrame(grid, columns=['grid'])
    data.insert(1, "ecdf", ecdf)
    data.insert(1, "Methods", method)
    sns.set()
    graph = sns.lineplot(x="grid",y="ecdf", style="Methods", color='red', palette="rocket", data=data, legend="brief")
    graph.set(title="Performance of Naive Pivots for Different Selection Methods", xlabel=" ",ylabel=" ")
    graph.plot([0, 0], [1, 1], linewidth=1, color="blue")
    plt.show()
    #plt.clf()
    #ecdf_MLE = ECDF(np.asarray(_pivot))
    #ecdf_MLE_adaptive = ECDF(np.asarray(_pivot_adaptive))
    #grid = np.linspace(0, 1, 101)
    #plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^', label="LASSO")
    #plt.plot(grid, ecdf_MLE_adaptive(grid), c='red', marker='o', label="Adaptive LASSO")
    #plt.plot(grid, grid, 'k--')
    #plt.title("Performance of Naive Pivots for Different Selection Methods")
    #plt.legend()
    #plt.show()

#ECDF_pivot_naive(nsim=50)


from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def plotPivot(pivot):
    robjects.r("""
    
               pivot_plot <- function(pivot, outpath='/Users/psnigdha/Research/Pivot_selective_MLE/ArXiV-2/submission-revision/', resolution=350, height=10, width=10)
               {
                    pivot = as.vector(pivot)
                    outfile = paste(outpath, 'pivot_LASSO_n200_gaussian_snr20.png', sep="")
                    png(outfile, res = resolution, width = width, height = height, units = 'cm')
                    par(mar=c(5,4,2,2)+0.1)
                    plot(ecdf(pivot), lwd=8, lty = 2, col="#000080", main="Model-4", ylab="", xlab="", cex.main=0.95)
                    abline(a = 0, b = 1, lwd=5, col="black")
                    dev.off()
               }                       
               """)

    R_plot = robjects.globalenv['pivot_plot']
    r_pivot = robjects.r.matrix(pivot, nrow=pivot.shape[0], ncol=1)
    R_plot(r_pivot)


def plotPivot_randomization(pivot_carved, pivot_randomized):
    robjects.r("""

               pivot_plot <- function(pivot_carved, pivot_randomized, 
               outpath='/Users/psnigdha/Research/Pivot_selective_MLE/ArXiV-2/submission-revision/', resolution=350, height=10, width=10)
               {
                    pivot_carved = as.vector(pivot_carved)
                    pivot_randomized = as.vector(pivot_randomized)
                    outfile = paste(outpath, 'randomized_pivot_LASSO_n100p1000_laplace_snr15.png', sep="")
                    png(outfile, res = resolution, width = width, height = height, units = 'cm')
                    par(mar=c(5,4,2,2)+0.1)
                    plot(ecdf(pivot_randomized), lwd=8, lty = 2, col="#000080", main="Model-4", ylab="", xlab="", cex.main=0.95)
                    plot(ecdf(pivot_carved), lwd=3, verticals=TRUE, add=TRUE, col='darkred')                    
                    abline(a = 0, b = 1, lwd=5, col="black")
                    dev.off()
               }                       
               """)

    R_plot = robjects.globalenv['pivot_plot']
    r_pivot_carved = robjects.r.matrix(pivot_carved, nrow=pivot_carved.shape[0], ncol=1)
    r_pivot_randomized = robjects.r.matrix(pivot_randomized, nrow=pivot_randomized.shape[0], ncol=1)
    R_plot(r_pivot_carved, r_pivot_randomized)

def main(nsim=200):
    _pivot=[]
    for i in range(nsim):
        _pivot.extend(test_approx_pivot_carved(n= 300,
                                               p= 500,
                                               signal_fac= 1.,
                                               s= 10,
                                               sigma= 1.,
                                               rho= 0.20,
                                               split_proportion=0.50))
        print("iteration completed ", i)

    plotPivot(np.asarray(_pivot))

#main()

def compare_pivots_highD(nsim=200):
    _carved_pivot = []
    _randomized_pivot = []

    for i in range(nsim):
        _carved_pivot.extend(test_approx_pivot_carved(n= 100,
                                                      p= 1000,
                                                      signal_fac= 0.6,
                                                      s= 10,
                                                      sigma= 1.,
                                                      rho= 0.40,
                                                      split_proportion=0.50))

        _randomized_pivot.extend(test_approx_pivot(n= 100,
                                                   p= 1000,
                                                   signal_fac= 0.6,
                                                   s= 10,
                                                   sigma= 1.,
                                                   rho= 0.40,
                                                   randomizer_scale= 1.))

        print("iteration completed ", i)

    plotPivot_randomization(np.asarray(_carved_pivot), np.asarray(_randomized_pivot))

#compare_pivots_highD(nsim=250)
