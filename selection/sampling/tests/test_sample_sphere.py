from __future__ import absolute_import
import nose
import nose.tools as nt
import numpy as np
import numpy.testing.decorators as dec

import matplotlib.pyplot as plt
import statsmodels.api as sm 

from scipy.stats import chi
import nose.tools as nt
import selection.constraints.affine as AC
from selection.algorithms.sqrt_lasso import sqrt_lasso, choose_lambda
from selection.distributions.discrete_family import discrete_family
from selection.tests.decorators import set_sampling_params_iftrue, set_seed_for_test

# generate a cone from a sqrt_lasso problem

def _generate_constraints(n=15, p=10, sigma=1):
    while True:
        y = np.random.standard_normal(n) * sigma
        beta = np.zeros(p)
        X = np.random.standard_normal((n,p)) + 0.3 * np.random.standard_normal(n)[:,None]
        X /= (X.std(0)[None,:] * np.sqrt(n))
        y += np.dot(X, beta) * sigma
        lam_theor = 0.3 * choose_lambda(X, quantile=0.9)
        L = sqrt_lasso(y, X, lam_theor)
        L.fit(tol=1.e-12, min_its=150)

        con = L.active_constraints
        if con is not None and L.active.shape[0] >= 3:
            break
    con.covariance = np.identity(con.covariance.shape[0])
    con.mean *= 0
    return con, y, L

@set_seed_for_test()
@set_sampling_params_iftrue(True)
def test_sample_ball(burnin=1000,
                       ndraw=1000,
                       nsim=None):

    p = 10
    A = np.identity(10)[:3]
    b = np.ones(3)
    initial = np.zeros(p)
    eta = np.ones(p)

    bound = 5
    s = AC.sample_truncnorm_white_ball(A,
                                       b, 
                                       initial,
                                       eta,
                                       lambda state: bound + 0.01 * np.random.sample() * np.linalg.norm(state)**2,
                                       burnin=burnin,
                                       ndraw=ndraw,
                                       how_often=5)
    return s

@set_seed_for_test()
@set_sampling_params_iftrue(True)
def test_sample_sphere(burnin=1000,
                       ndraw=1000,
                       nsim=None):

    p = 10
    A = np.identity(10)[:3]
    b = 2 * np.ones(3)
    mean = -np.ones(p)
    noise = np.random.standard_normal(p) * 0.1
    noise[-3:] = 0.
    initial = noise + mean
    eta = np.ones(p)

    bound = 5
    s1 = AC.sample_truncnorm_white_sphere(A,
                                          b, 
                                          initial,
                                          eta,
                                          how_often=20,
                                          burnin=burnin,
                                          ndraw=ndraw)

    con = AC.constraints(A, b)
    con.covariance = np.diag([1]*7 + [0]*3)
    con.mean[:] = mean
    s2 = AC.sample_from_sphere(con, initial, ndraw=ndraw, burnin=burnin)
    return s1, s2

@dec.slow
@set_seed_for_test(20)
@set_sampling_params_iftrue(True, nsim=50)
def test_distribution_sphere(n=15, p=10, sigma=1.,
                             nsim=2000,
                             sample_constraints=False,
                             burnin=10000,
                             ndraw=10000):

    # see if we really are sampling from 
    # correct distribution
    # by comparing to an accept-reject sampler

    con, y = _generate_constraints()[:2]
    accept_reject_sample = []

    hit_and_run_sample, W = AC.sample_from_sphere(con, y, 
                                                  ndraw=ndraw,
                                                  burnin=burnin)
    statistic = lambda x: np.fabs(x).max()
    family = discrete_family([statistic(s) for s in hit_and_run_sample], W)
    radius = np.linalg.norm(y)

    count = 0

    pvalues = []

    while True:

        U = np.random.standard_normal(n)
        U /= np.linalg.norm(U)
        U *= radius

        if con(U):
            accept_reject_sample.append(U)
            count += 1

            true_sample = np.array([statistic(s) for s in accept_reject_sample])

            if (count + 1) % int(nsim / 10) == 0:

                pvalues.extend([family.cdf(0, t) for t in true_sample])
                print np.mean(pvalues), np.std(pvalues)

                if sample_constraints:
                    con, y = _generate_constraints()[:2]

                hit_and_run_sample, W = AC.sample_from_sphere(con, y, 
                                                              ndraw=ndraw,
                                                              burnin=burnin)
                family = discrete_family([statistic(s) for s in hit_and_run_sample], W)
                radius = np.linalg.norm(y)
                accept_reject_sample = []

        if count >= nsim:
            break

    U = np.linspace(0, 1, 101)
    plt.plot(U, sm.distributions.ECDF(pvalues)(U))
    plt.plot([0,1],[0,1])

@set_seed_for_test()
@set_sampling_params_iftrue(True)
def test_conditional_sampling(n=20, p=25, sigma=20,
                              ndraw=1000,
                              burnin=1000,
                              nsim=None):
    """
    goodness of fit samples from
    inactive constraints intersect a sphere

    this test verifies the sampler is doing what it should
    """

    con, y, L = _generate_constraints(n=n, p=p, sigma=sigma)

    con = L.inactive_constraints
    conditional_con = con.conditional(L._X_E.T, np.dot(L._X_E.T, y))

    Z, W = AC.sample_from_sphere(conditional_con, 
                                 y,
                                 ndraw=ndraw,
                                 burnin=burnin)  
    
    T1 = np.dot(L._X_E.T, Z.T) - np.dot(L._X_E.T, y)[:,None]
    nt.assert_true(np.linalg.norm(T1) < 1.e-7)

    T2 = (np.dot(L.R_E, Z.T)**2).sum(0) - np.linalg.norm(np.dot(L.R_E, y))**2
    nt.assert_true(np.linalg.norm(T2) < 1.e-7)
