import numpy as np
from base import selective_loss
from regreg.smooth.glm import logistic_loss


class logistic_Xrandom_new(selective_loss):

    def __init__(self, X, y,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 initial=None):
        selective_loss.__init__(self, X.shape[1],
                                coef=coef,
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial)

        self.X = X.copy()
        self.y = y.copy()
        #self._restricted_grad_beta = np.zeros(self.shape)
        #self._cov = np.dot(X.T,X)

    def smooth_objective(self, beta, mode='both',
                         check_feasibility=False):

        _loss = logistic_loss(self.X, self.y, coef=1.)

        return _loss.smooth_objective(beta, mode=mode, check_feasibility=check_feasibility)

    # this is something that regreg does not know about, i.e.
    # what is data and what is not...

    def fit_restricted(self, active, solve_args={'min_its':50, 'tol':1.e-10}):
        """
        Fits the logistic regression to a candidate active set, without penalty.
        Calls the method bootstrap_covariance() to bootstrap the covariance matrix.

        Parameters:
        ----------

        active: np.bool
            The active set from fitting the logistic lasso

        solve_args: dict
            Arguments to be passed to regreg solver.

        Returns:
        --------

        None

        Notes:
        ------

        Sets self._beta_unpenalized which will be used in the covariance matrix calculation.
        Also computes Hessian of logistic loss as well as the bootstrap covariance.

        """

        self.active = active
        self.size_active = np.sum(self.active)

        if self.active.any():
            self.inactive = ~active
            X_E = self.X[:, self.active]
            loss_E = logistic_loss(X_E, self.y)
            self._beta_unpenalized = loss_E.solve(**solve_args)
            self.hessian()
            self._restricted_hessian = self._hessian[:, self.active]
            self.bootstrap_covariance()
        else:
            raise ValueError("Empty active set.")


    def bootstrap_covariance(self):
        """
        """
        if not hasattr(self, "_beta_unpenalized"):
            raise ValueError("method fit_restricted has to be called before computing the covariance")

        if not hasattr(self, "_cov"):

            X, y = self.X, self.y
            n, p = X.shape
            nsample = 2000
            active=self.active
            inactive=~active
            nactive = self.size_active

            def pi(X):
                w = np.exp(np.dot(X[:,self.active], self._beta_unpenalized))
                return w / (1 + w)

            _mean_cum = 0

            self._cov = np.zeros((p,p))
            Q = np.zeros((p,p))

            pi_E = pi(X)
            W = np.diag(np.diag(np.outer(pi_E, 1-pi_E)))
            Q = np.dot(X[:,active].T, np.dot(W, X[:, active]))
            Q_inv = np.linalg.inv(Q)
            C = np.dot(X[:,inactive].T, np.dot(W, X[:, active]))
            I = np.dot(C,Q_inv)

            for _ in range(nsample):
                indices = np.random.choice(n, size=(n,), replace=True)
                y_star = y[indices]
                X_star = X[indices]
                pi_star = pi(X_star)
                #print 'pi size', pi_star.shape
                Z_star_active = np.dot(X_star[:, active].T, y_star - pi_star)
                Z_star_inactive = np.dot(X_star[:, inactive].T, y_star-pi_star)

                Z_1 = np.dot(Q_inv, Z_star_active)
                Z_2 = Z_star_inactive + np.dot(I, Z_star_active)
                Z_star = np.concatenate((Z_1, Z_2), axis=0)

                _mean_cum += Z_star
                self._cov += np.multiply.outer(Z_star, Z_star)


            self._cov /= float(nsample)
            _mean = _mean_cum / float(nsample)
            self._cov -= np.multiply.outer(_mean, _mean)

            return self._cov

    @property
    def covariance(self, doc="Covariance of sufficient statistic $X^Ty$."):
        if not hasattr(self, "_cov"):
            self.bootstrap_covariance()
        return self._cov

    def gradient(self, data, beta):
        """
        Gradient of smooth part restricted to active set
        """

        #if not hasattr(self, "_cov"):
        #    self.bootstrap_covariance()

        #g = -(data - np.dot(self._cov, beta))
        data1 = data.copy()

        data1[:self.size_active] = 0  # last p-|E| coordinates of data vector kept, first |E| become zeros
        # (0, N), N is the null statistic, N=X_{-E}^Y(y-X_E\bar{\beta}_E)


        restricted_hessian = self._restricted_hessian
        #restricted_hessian = self._cov_inv[:, :np.sum(self.active)]

        #g = - data1 + np.dot(restricted_hessian, beta[self.active] - data[:self.size_active])
        g = np.dot(restricted_hessian, beta[self.active] - data[:self.size_active])
        g[~self.active] -= data1[self.size_active:]
        return g


    def hessian(self):
        """
        hessian is constant in this case.
        """
        #if not hasattr(self, "_cov"):
        #    self.bootstrap_covariance()

        def pi_beta(X, beta):
            w = np.exp(np.dot(X[:, self.active], beta))
            return w / (1 + w)

        _pi = pi_beta(self.X, self._beta_unpenalized)  # n-dim
        W = np.diag(np.diag(np.outer(_pi, 1-_pi)))

        self._hessian = np.dot(self.X.T,np.dot(W, self.X))

        print "hessian diag", np.diag(np.linalg.inv(self._hessian[self.active][:, self.active]))

        return np.dot(self.X.T,np.dot(W, self.X))





