import numpy as np
from selection.algorithms.softmax import nonnegative_softmax
from selection.bayesian.selection_probability.rr import cube_subproblem_scaled, cube_barrier_scaled,\
    cube_gradient_scaled, cube_hessian_scaled, cube_objective
import regreg.api as rr

class selection_probability_objective_ms(rr.smooth_atom):
    def __init__(self,
                 T,
                 feasible_point,
                 active,
                 active_signs,
                 threshold,
                 mean_parameter,  # in R^n
                 noise_var_covar, #a matrix this time
                 randomizer,
                 epsilon = 0.,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        """
        Objective function for $\beta_E$ (i.e. active) with $E$ the `active_set` optimization
        variables, and data $z \in \mathbb{R}^n$ (i.e. response).
        NEEDS UPDATING
        Above, $\beta_E^*$ is the `parameter`, $b_{\geq}$ is the softmax of the non-negative constraint,
        $$
        B_E = X^TX_E
        $$
        and
        $$
        \gamma_E = \begin{pmatrix} \lambda s_E\\ 0\end{pmatrix}
        $$
        with $\lambda$ being `lagrange`.
        Parameters
        ----------
        X : np.float
             Design matrix of shape (n,p)
        active : np.bool
             Boolean indicator of active set of shape (p,).
        active_signs : np.float
             Signs of active coefficients, of shape (active.sum(),).
        lagrange : np.float
             Array of lagrange penalties for LASSO of shape (p,)
        parameter : np.float
             Parameter $\beta_E^*$ for which we want to
             approximate the selection probability.
             Has shape (active_set.sum(),)
        randomization : np.float
             Variance of IID Gaussian noise
             that was added before selection.
        """

        p = T.shape
        E = active.sum()
        self.active = active

        w, v = np.linalg.eig(noise_var_covar)
        var_half_inv = (v.T.dot(np.diag(1./w))).dot(v)
        self.randomization = randomizer

        self.inactive_conjugate = self.active_conjugate = randomizer.CGF_conjugate
        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        #self.inactive_threshold = threshold[~active]

        initial = np.zeros(p + E, )
        initial[p:] = feasible_point

        rr.smooth_atom.__init__(self,
                                (p + E,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        nonnegative = nonnegative_softmax(E)

        opt_vars = np.zeros(p + E, bool)
        opt_vars[p:] = 1

        self._opt_selector = rr.selector(opt_vars, (p + E,))
        self.nonnegative_barrier = nonnegative.linear(self._opt_selector)
        self._response_selector = rr.selector(~opt_vars, (p + E,))

        self.A_active = np.hstack([-np.identity(E), np.zeros((E,p-E)), np.diag(threshold[active])* active_signs[None, :]])
        self.A_inactive = np.hstack([np.zeros((p-E,E)), np.identity(p-E), np.zeros(E)])

        # defines \gamma and likelihood loss
        self.set_parameter(mean_parameter, var_half_inv)

        self.active_conj_loss = rr.affine_smooth(self.active_conjugate, self.A_active)

        cube_obj = cube_objective(self.inactive_conjugate,
                                  threshold[~active],
                                  nstep=nstep)

        self.cube_loss = rr.affine_smooth(cube_obj, self.A_inactive)

        self.total_loss = rr.smooth_sum([self.active_conj_loss,
                                         self.cube_loss,
                                         self.likelihood_loss,
                                         self.nonnegative_barrier])

    def set_parameter(self, mean_parameter, var_half_inv):
        """
        Set $\beta_E^*$.
        """
        mean = var_half_inv.dot(np.squeeze(mean_parameter))
        likelihood_loss = rr.signal_approximator(mean, coef=1.)
        _likelihood_loss = rr.affine_smooth(likelihood_loss, self._response_selector)
        self.likelihood_loss = rr.affine_smooth(_likelihood_loss, var_half_inv)

    def smooth_objective(self, param, mode='both', check_feasibility=False):
        """
        Evaluate the smooth objective, computing its value, gradient or both.
        Parameters
        ----------
        mean_param : ndarray
            The current parameter values.
        mode : str
            One of ['func', 'grad', 'both'].
        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `mean_param` is not
            in the domain.
        Returns
        -------
        If `mode` is 'func' returns just the objective value
        at `mean_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """

        param = self.apply_offset(param)

        if mode == 'func':
            f = self.total_loss.smooth_objective(param, 'func')
            return self.scale(f)
        elif mode == 'grad':
            g = self.total_loss.smooth_objective(param, 'grad')
            return self.scale(g)
        elif mode == 'both':
            f, g = self.total_loss.smooth_objective(param, 'both')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def minimize(self, initial=None, min_its=10, max_its=50, tol=1.e-10):

        nonneg_con = self._opt_selector.output_shape[0]
        constraint = rr.separable(self.shape,
                                  [rr.nonnegative((nonneg_con,), offset=1.e-12 * np.ones(nonneg_con))],
                                  [self._opt_selector.index_obj])

        problem = rr.separable_problem.fromatom(constraint, self)
        problem.coefs[:] = 0.5
        soln = problem.solve(max_its=max_its, min_its=min_its, tol=tol)
        value = problem.objective(soln)
        return soln, value

    def minimize2(self, step=1, nstep=30, tol=1.e-8):

        n, p = self._X.shape

        current = self.coefs
        current_value = np.inf

        objective = lambda u: self.smooth_objective(u, 'func')
        grad = lambda u: self.smooth_objective(u, 'grad')

        for itercount in range(nstep):
            newton_step = grad(current) * self.noise_variance

            # make sure proposal is feasible

            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                if np.all(proposal[n:] > 0):
                    break
                step *= 0.5
                if count >= 40:
                    raise ValueError('not finding a feasible point')

            # make sure proposal is a descent

            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)
                # print(current_value, proposed_value, 'minimize')
                if proposed_value <= current_value:
                    break
                step *= 0.5

            # stop if relative decrease is small

            if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
                current = proposal
                current_value = proposed_value
                break

            current = proposal
            current_value = proposed_value

            if itercount % 4 == 0:
                step *= 2

        #print('iter', itercount)
        value = objective(current)
        return current, value