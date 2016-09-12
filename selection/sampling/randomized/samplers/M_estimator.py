import numpy as np
import regreg.api as rr

class M_estimator(object):

    def __init__(self, loss, epsilon, penalty, randomization, solve_args={'min_its':50, 'tol':1.e-10}):
        """
        Fits the logistic regression to a candidate active set, without penalty.
        Calls the method bootstrap_covariance() to bootstrap the covariance matrix.
        Computes $\bar{\beta}_E$ which is the restricted
        M-estimator (i.e. subject to the constraint $\beta_{-E}=0$).
        Parameters:
        -----------
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
        Also computes Hessian of loss at restricted M-estimator as well as the bootstrap covariance.
        """

        (self.loss,
         self.epsilon,
         self.penalty,
         self.randomization,
         self.solve_args) = (loss,
                             epsilon,
                             penalty,
                             randomization,
                             solve_args)

    def solve(self):

        (loss,
         epsilon,
         penalty,
         randomization,
         solve_args) = (self.loss,
                        self.epsilon,
                        self.penalty,
                        self.randomization,
                        self.solve_args)

        # initial solution

        problem = rr.simple_problem(loss, penalty)
        self._randomZ = self.randomization.sample()
        self._random_term = rr.identity_quadratic(epsilon, 0, -self._randomZ, 0)
        self.initial_soln = problem.solve(self._random_term, **solve_args)

    def setup_sampler(self, solve_args={'min_its':50, 'tol':1.e-10}):

        (loss,
         epsilon,
         penalty,
         randomization,
         initial_soln) = (self.loss,
                          self.epsilon,
                          self.penalty,
                          self.randomization,
                          self.initial_soln)

        # find the active groups and their direction vectors
        # as well as unpenalized groups

        groups = np.unique(penalty.groups)
        active_groups = np.zeros(len(groups), np.bool)
        unpenalized_groups = np.zeros(len(groups), np.bool)

        active_directions = []
        active = np.zeros(loss.shape, np.bool)
        unpenalized = np.zeros(loss.shape, np.bool)

        initial_scalings = []

        for i, g in enumerate(groups):
            group = penalty.groups == g
            active_groups[i] = (np.linalg.norm(initial_soln[group]) > 1.e-6 * penalty.weights[g]) and (penalty.weights[g] > 0)
            unpenalized_groups[i] = (penalty.weights[g] == 0)
            if active_groups[i]:
                active[group] = True
                z = np.zeros(active.shape, np.float)
                z[group] = initial_soln[group] / np.linalg.norm(initial_soln[group])
                active_directions.append(z)
                initial_scalings.append(np.linalg.norm(initial_soln[group]))
            if unpenalized_groups[i]:
                unpenalized[group] = True

        # solve the restricted problem

        overall = active + unpenalized
        inactive = ~overall

        # initial state for opt variables

        initial_subgrad = -(self.loss.smooth_objective(self.initial_soln, 'grad') + self._random_term.objective(self.initial_soln, 'grad') + epsilon * self.initial_soln)
        initial_subgrad = initial_subgrad[inactive]
        initial_unpenalized = self.initial_soln[unpenalized]
        self.observed_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized,
                                                  initial_subgrad], axis=0)

        active_directions = np.array(active_directions).T

        # we are implicitly assuming that
        # loss is a pairs model

        _beta_unpenalized = restricted_Mest(loss, overall, solve_args=solve_args)

        beta_full = np.zeros(active.shape)
        beta_full[overall] = _beta_unpenalized
        _hessian = loss.hessian(beta_full)
        self._beta_full = beta_full

        # observed state for score

        self.observed_score_state = np.hstack([_beta_unpenalized,
                                               -loss.smooth_objective(beta_full, 'grad')[inactive]])

        # form linear part

        self.num_opt_var = p = loss.shape[0] # shorthand for p

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        _opt_linear_term = np.zeros((p, p))
        _score_linear_term = np.zeros((p, active_groups.sum() + unpenalized.sum() + inactive.sum()))

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator

        Mest_slice = slice(0, overall.sum())
        _Mest_hessian = _hessian[:,overall]
        _score_linear_term[:,Mest_slice] = -_Mest_hessian

        # N_{-(E \cup U)} piece -- inactive coordinates of score of M estimator at unpenalized solution

        null_slice = slice(overall.sum(), p)
        _score_linear_term[inactive][:,null_slice] = -np.identity(inactive.sum())

        # c_E piece

        scaling_slice = slice(0, active_groups.sum())
        _opt_hessian = (_hessian + epsilon * np.identity(p)).dot(active_directions)
        _opt_linear_term[:,scaling_slice] = _opt_hessian

        # beta_U piece

        unpenalized_slice = slice(active_groups.sum(), active_groups.sum() + unpenalized.sum())
        unpenalized_directions = np.identity(p)[:,unpenalized]
        if unpenalized.sum():
            _opt_linear_term[:,unpenalized_slice] = (_hessian + epsilon * np.identity(p)).dot(unpenalized_directions)

        # subgrad piece
        subgrad_slice = slice(active_groups.sum() + unpenalized.sum(), active_groups.sum() + inactive.sum() + unpenalized.sum())
        _opt_linear_term[inactive][:,subgrad_slice] = np.identity(inactive.sum())

        # form affine part

        _opt_affine_term = np.zeros(p)
        idx = 0
        for i, g in enumerate(groups):
            if active_groups[i]:
                group = penalty.groups == g
                _opt_affine_term[group] = active_directions[:,idx][group] * penalty.weights[g]
                idx += 1

        # two transforms that encode score and optimization
        # variable roles

        # later, conditioning will modify `score_transform`

        self.opt_transform = (_opt_linear_term, _opt_affine_term)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        # now store everything needed for the projections
        # the projection acts only on the optimization
        # variables

        self.scaling_slice = scaling_slice

        new_groups = penalty.groups[inactive]
        new_weights = dict([(g,penalty.weights[g]) for g in penalty.weights.keys() if g in np.unique(new_groups)])

        # we form a dual group lasso object
        # to do the projection

        self.group_lasso_dual = rr.group_lasso_dual(new_groups, weights=new_weights, bound=1.)
        self.subgrad_slice = subgrad_slice

        # store active sets, etc.

        (self.overall,
         self.active,
         self.unpenalized,
         self.inactive) = (overall,
                           active,
                           unpenalized,
                           inactive)

    def projection(self, opt_state):
        """
        Full projection for Langevin.
        The state here will be only the state of the optimization variables.
        """

        if not hasattr(self, "scaling_slice"):
            raise ValueError('setup_sampler should be called before using this function')

        new_state = opt_state.copy() # not really necessary to copy
        new_state[self.scaling_slice] = np.maximum(opt_state[self.scaling_slice], 0)
        new_state[self.subgrad_slice] = self.group_lasso_dual.bound_prox(opt_state[self.subgrad_slice])

        return new_state

    def gradient(self, data_state, data_transform, opt_state):
        """
        Randomization derivative at full state.
        """

        if not hasattr(self, "opt_transform"):
            raise ValueError('setup_sampler should be called before using this function')

        # omega
        opt_linear, opt_offset = self.opt_transform
        data_linear, data_offset = data_transform
        data_piece = data_linear.dot(data_state) + data_offset
        opt_piece = opt_linear.dot(opt_state) + opt_offset
        full_state = (data_piece + opt_piece)
        randomization_derivative = self.randomization.gradient(full_state)
        data_grad = data_linear.T.dot(randomization_derivative)
        opt_grad = opt_linear.T.dot(randomization_derivative)
        return data_grad, opt_grad


    def condition(self, target_score_cov, target_cov, observed_target_state):
        """
        condition the score on the target,
        return a new score_transform
        that is composition of `self.score_transform`
        with the affine map from conditioning
        """

        target_score_cov = np.atleast_2d(target_score_cov)
        target_cov = np.atleast_2d(target_cov)
        observed_target_state = np.atleast_1d(observed_target_state)

        linear_part = target_score_cov.T.dot(np.linalg.pinv(target_cov))
        offset = self.observed_score_state - linear_part.dot(observed_target_state)

        # now compute the composition of this map with
        # self.score_transform

        score_linear, score_offset = self.score_transform
        composition_linear_part = score_linear.dot(linear_part)
        composition_offset = score_linear.dot(offset) + score_offset

        return (composition_linear_part, composition_offset)

def restricted_Mest(Mest_loss, active, solve_args={'min_its':50, 'tol':1.e-10}):

    X, Y = Mest_loss.data

    if Mest_loss._is_transform:
        raise NotImplementedError('to fit restricted model, X must be an ndarray or scipy.sparse; general transforms not implemented')
    X_restricted = X[:,active]
    loss_restricted = rr.affine_smooth(Mest_loss.saturated_loss, X_restricted)
    beta_E = loss_restricted.solve(**solve_args)

    return beta_E
