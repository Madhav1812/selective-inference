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

    def fit_E(self, active, solve_args={'min_its':50, 'tol':1.e-10}):
        """
        Fits the logistic regression after seeing the active set, without penalty.
        Calls the method bootstrap_covariance() to bootstrap the covariance matrix.

        Parameters:
        ----------
        active: the active set from fitting the logistic lasso

        solve_args: passed to regreg.simple_problem.solve

        Returns:
        --------
        Set self._beta_unpenalized which will be used in the covariance matrix calculation.
        """

        self.active = active
        #print self.active
        self.size_active = np.sum(self.active)
        #print self.size_active
        if self.active.any():
            self.inactive = ~active
            X_E = self.X[:, self.active]
            loss_E = logistic_loss(X_E, self.y)
            self._beta_unpenalized = loss_E.solve(**solve_args)
            self.hessian()

            self.bootstrap_covariance()
        else:
            raise ValueError("Empty active set.")


    def bootstrap_covariance(self):
        """
        """
        if not hasattr(self, "_beta_unpenalized"):
            raise ValueError("method fit_E has to be called before computing the covariance")

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
            #self._cov_inv = np.linalg.inv(self._cov)


            #self._cov_beta_bar = self._cov[:nactive][:,:nactive]
            #self._inv_cov_beta_bar = np.linalg.inv(self._cov_beta_bar)

            #print "beta_bar ", np.diag(self._cov_beta_bar)
            #self._cov_T = np.dot(self.X[:, self.active].T, np.dot(W, self.X[:, self.active]))
            #self._cov_T_inv = np.linalg.inv(self._cov_T)
            #self.L = np.linalg.cholesky(self._cov)

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


        restricted_hessian = self.hessian[:, self.active]
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

        self.hessian = np.dot(self.X.T,np.dot(W, self.X))

        print "hessian diag", np.diag(np.linalg.inv(self.hessian[self.active][:, self.active]))

        return np.dot(self.X.T,np.dot(W, self.X))


    def setup_sampling(self, data, mean, linear_part, value):
        """
        Set up the sampling conditioning on the KKT constraints as well as
        the linear constraints C * data = d

        Parameters:
        ----------
        data:
        The subject of the sampling. In this case the gradient of loss at 0.

        mean: \beta^0_E

        sigma: default to None in logistic lasso

        linear_part: C

        value: d
        """

        self.accept_data = 0
        self.total_data = 0

        P = np.dot(linear_part.T, np.linalg.pinv(linear_part).T)
        I = np.identity(linear_part.shape[1])


        self.data = data
        self.mean = mean


        self.R = I - P
        #print 'nonzeros in R', np.count_nonzero(self.R)
        self.P = P
        self.linear_part = linear_part


    def proposal(self, data):
        if not hasattr(self, "L"):
            self.bootstrap_covariance()

        n, p = self.X.shape
        stepsize = 30. / np.sqrt(p)
        #new = data + stepsize * np.dot(self.R,
        #                               np.dot(self.L, np.random.standard_normal(p)))

        new = data + stepsize * np.dot(self.R,
                                       np.random.standard_normal(p))
        #print 'data differ',  np.count_nonzero(data-new)
        log_transition_p = self.logpdf(new) - self.logpdf(data)
        return new, log_transition_p

    def logpdf(self, data):
        return -((data-self.mean)*np.dot(self._cov_inv, data-self.mean)).sum() / 2

    def update_proposal(self, state, proposal, logpdf):
        pass








