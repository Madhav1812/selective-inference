import numpy as np
import numpy.testing.decorators as dec

from selection.algorithms.lasso import (lasso, 
                                        data_carving, 
                                        instance, 
                                        split_model, 
                                        instance, 
                                        nominal_intervals)

from selection.tests.decorators import set_sampling_params_iftrue

def test_class(n=100, p=20):
    y = np.random.standard_normal(n)
    X = np.random.standard_normal((n,p))
    lam_theor = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 1000)))).max(0))
    L = lasso(y,X,lam=0.5*lam_theor)
    L.fit()
    L.form_constraints()
    C = L.constraints

    np.testing.assert_array_less( \
        np.dot(L.constraints.linear_part, L.y),
        L.constraints.offset)

    I = L.intervals
    P = L.active_pvalues

    return L, C, I, P

@set_sampling_params_iftrue(True)
def test_data_carving(n=100,
                      p=200,
                      s=7,
                      sigma=5,
                      rho=0.3,
                      snr=7.,
                      split_frac=0.9,
                      lam_frac=2.,
                      ndraw=8000,
                      burnin=2000, 
                      df=np.inf,
                      coverage=0.90,
                      compute_intervals=True,
                      nsim=None):

    counter = 0

    return_value = []

    while True:
        counter += 1
        X, y, beta, active, sigma = instance(n=n, 
                                             p=p, 
                                             s=s, 
                                             sigma=sigma, 
                                             rho=rho, 
                                             snr=snr, 
                                             df=df)
        mu = np.dot(X, beta)
        L, stage_one = split_model(y, X, 
                        sigma=sigma,
                        lam_frac=lam_frac,
                        split_frac=split_frac)[:2]

        if set(range(s)).issubset(L.active):
            while True:
                results, L = data_carving(y, X, lam_frac=lam_frac, 
                                          sigma=sigma,
                                          stage_one=stage_one,
                                          splitting=True, 
                                          ndraw=ndraw,
                                          burnin=burnin,
                                          coverage=coverage,
                                          compute_intervals=compute_intervals)
                if set(range(s)).issubset(L.active):
                    print "succeed"
                    break
                print "failed at least once"

            carve = [r[1] for r in results]
            split = [r[3] for r in results]

            Xa = X[:,L.active]
            truth = np.dot(np.linalg.pinv(Xa), mu) 

            split_coverage = []
            carve_coverage = []
            for result, t in zip(results, truth):
                _, _, ci, _, si = result
                carve_coverage.append((ci[0] < t) * (t < ci[1]))
                split_coverage.append((si[0] < t) * (t < si[1]))

            TP = s
            FP = L.active.shape[0] - TP
            v = (carve[s:], split[s:], carve[:s], split[:s], counter, carve_coverage, split_coverage, TP, FP)
            return_value.append(v)
            break
        else:
            TP = len(set(L.active).intersection(range(s)))
            FP = L.active.shape[0] - TP
            v = (None, None, None, None, counter, np.nan, np.nan, TP, FP)
            return_value.append(v)
    return return_value

@set_sampling_params_iftrue(True)
@dec.skipif(True, "needs a data_carving_coverage function to be defined")
def test_data_carving_coverage(nsim=200, 
                               coverage=0.8,
                               ndraw=8000,
                               burnin=2000):
    C = []
    SE = np.sqrt(coverage * (1 - coverage) / nsim)

    while True:
        C.extend(data_carving_coverage(ndraw=ndraw, burnin=burnin)[-1])
        if len(C) > nsim:
            break

    if np.fabs(np.mean(C) - coverage) > 3 * SE:
        raise ValueError('coverage not within 3 SE of where it should be')

    return C

def test_intervals(n=100, p=20, s=5):
    t = []
    X, y, beta = instance(n=n, p=p, s=s)[:3]
    las = lasso(y, X, 4., sigma = .25)
    las.fit()
    las.form_constraints()

    # smoke test

    las.soln
    las.active_constraints
    las.inactive_constraints
    las.constraints
    las.active_pvalues
    intervals = las.intervals
    nominal_intervals(las)
    t.append([(beta[I], L, U) for I, L, U in intervals])
    return t
    
