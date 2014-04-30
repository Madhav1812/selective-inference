"""
This module contains the core code needed for post selection
inference based on affine selection procedures as
described in the papers `Kac Rice`_, `Spacings`_, `covTest`_
and `post selection LASSO`_.

.. _covTest: http://arxiv.org/abs/1301.7161
.. _Kac Rice: http://arxiv.org/abs/1308.3020
.. _Spacings: http://arxiv.org/abs/1401.3889
.. _post selection LASSO: http://arxiv.org/abs/1311.6238

"""

import numpy as np
from .pvalue import truncnorm_cdf, norm_interval
from .truncated import truncated_gaussian
from .sample_truncnorm import sample_truncnorm_white, sample_truncnorm_white_sphere
                        
from warnings import warn

WARNINGS = False

class constraints(object):

    r"""
    This class is the core object for affine selection procedures.
    It is meant to describe sets of the form $C$
    where

    .. math::

       C = \left\{z: Az\leq b \right \}

    Its main purpose is to consider slices through $C$
    and the conditional distribution of a Gaussian $N(\mu,\Sigma)$
    restricted to such slices.

    Notes
    -----

    In this parameterization, the parameter `self.mean` corresponds
    to the *reference measure* that is being truncated. It is not the
    mean of the truncated Gaussian.

    """

    def __init__(self, 
                 linear_part,
                 offset,
                 covariance=None,
                 mean=None):
        r"""
        Create a new inequality. 

        Parameters
        ----------

        linear_part : np.float((q,p))
            The linear part, $A$ of the affine constraint
            $\{z:Az \leq b\}$. 

        equality: (C,d)
            The offset part, $b$ of the affine constraint
            $\{z:Cz=d\}$. 

        covariance : np.float
            Covariance matrix of Gaussian distribution to be 
            truncated. Defaults to `np.identity(self.dim)`.

        mean : np.float
            Mean vector of Gaussian distribution to be 
            truncated. Defaults to `np.zeros(self.dim)`.

        """

        self.linear_part, self.offset = \
            np.asarray(linear_part), np.asarray(offset)
        if self.linear_part.ndim == 2:
            self.dim = self.linear_part.shape[1]
        else:
            self.dim = self.linear_part.shape[0]

        if covariance is None:
            covariance = np.identity(self.dim)
        self.covariance = covariance

        if mean is None:
            mean = np.zeros(self.dim)
        self.mean = mean

    def _repr_latex_(self):
        return """$$Z \sim N(\mu,\Sigma) | AZ \leq b$$"""

    def __call__(self, Y, tol=1.e-3):
        r"""
        Check whether Y satisfies the linear
        inequality and equality constraints.
        """
        V1 = np.dot(self.linear_part, Y) - self.offset
        return np.all(V1 < tol * np.fabs(V1).max())

    def conditional(self, linear_part, value):
        """
        Return an equivalent constraint with a
        after having conditioned on a linear equality.
        
        Let the inequality constraints be specified by
        `(A,b)` and the inequality constraints be specified
        by `(C,d)`. We form equivalent inequality constraints by 
        considering the residual

        .. math::
           
           AY - E(AY|CZ=d)

        """

        A, b, S = self.linear_part, self.offset, self.covariance
        C, d = linear_part, value

        M1 = np.dot(S, C.T)
        M2 = np.dot(C, M1)
        if M2.shape:
            M2i = np.linalg.pinv(M2)
            delta_cov = np.dot(M1, np.dot(M2i, M1.T))
            delta_offset = np.dot(M1, np.dot(M2i, d))
        else:
            M2i = 1. / M2
            delta_cov = np.multiply.outer(M1, M1) / M2i
            delta_offset = M1 * d  / M2i

        return constraints(self.linear_part,
                           self.offset - np.dot(self.linear_part, delta_offset),
                           covariance=self.covariance - delta_cov,
                           mean=self.mean)

    def bounds(self, direction_of_interest, Y):
        r"""
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$.

        Parameters
        ----------

        direction_of_interest: np.float
            A direction $\eta$ for which we may want to form 
            selection intervals or a test.

        Y : np.float
            A realization of $N(\mu,\Sigma)$ where 
            $\Sigma$ is `self.covariance`.

        Returns
        -------

        L : np.float
            Lower truncation bound.

        Z : np.float
            The observed $\eta^TY$

        U : np.float
            Upper truncation bound.

        S : np.float
            Standard deviation of $\eta^TY$.

        Notes
        -----
        
        This method assumes that equality constraints
        have been enforced and direction of interest
        is in the row space of any equality constraint matrix.
        
        """
        return interval_constraints(self.linear_part,
                                    self.offset,
                                    self.covariance,
                                    Y,
                                    direction_of_interest)

    def pivot(self, direction_of_interest, Y,
              alternative='greater'):
        r"""
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$ and test whether 
        $\eta^T\mu$ is greater then 0, less than 0 or equal to 0.

        Parameters
        ----------

        direction_of_interest: np.float
            A direction $\eta$ for which we may want to form 
            selection intervals or a test.

        Y : np.float
            A realization of $N(0,\Sigma)$ where 
            $\Sigma$ is `self.covariance`.

        alternative : ['greater', 'less', 'twosided']
            What alternative to use.

        Returns
        -------

        P : np.float
            $p$-value of corresponding test.

        Notes
        -----

        All of the tests are based on the exact pivot $F$ given
        by the truncated Gaussian distribution for the
        given direction $\eta$. If the alternative is 'greater'
        then we return $1-F$; if it is 'less' we return $F$
        and if it is 'twosided' we return $2 \min(F,1-F)$.

        This method assumes that equality constraints
        have been enforced and direction of interest
        is in the row space of any equality constraint matrix.
        
        """
        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")
        L, Z, U, S = self.bounds(direction_of_interest, Y)
        meanZ = (direction_of_interest * self.mean).sum()
        P = truncnorm_cdf((Z-meanZ)/S, (L-meanZ)/S, (U-meanZ)/S)
        if alternative == 'greater':
            return 1 - P
        elif alternative == 'less':
            return P
        else:
            return 2 * min(P, 1-P)

    def interval(self, direction_of_interest, Y,
                 alpha=0.05, UMAU=False):
        r"""
        For a realization $Y$ of the random variable $N(\mu,\Sigma)$
        truncated to $C$ specified by `self.constraints` compute
        the slice of the inequality constraints in a 
        given direction $\eta$ and test whether 
        $\eta^T\mu$ is greater then 0, less than 0 or equal to 0.

        Parameters
        ----------

        direction_of_interest: np.float

            A direction $\eta$ for which we may want to form 
            selection intervals or a test.

        Y : np.float

            A realization of $N(0,\Sigma)$ where 
            $\Sigma$ is `self.covariance`.

        alpha : float

            What level of confidence?

        UMAU : bool

            Use the UMAU intervals?

        Returns
        -------

        [U,L] : selection interval

        Notes
        -----
        
        This method assumes that equality constraints
        have been enforced and direction of interest
        is in the row space of any equality constraint matrix.
        
        """

        return selection_interval( \
            self.linear_part,
            self.offset,
            self.covariance,
            Y,
            direction_of_interest,
            alpha=alpha,
            UMAU=UMAU)

    def whiten(self):
        """
        Return a whitened version of constraints in a different
        basis, and a change of basis matrix.

        If `self.covariance` is rank deficient, the change-of
        basis matrix will not be square.

        """

        rank = np.linalg.matrix_rank(self.covariance)
        D, U = np.linalg.eigh(self.covariance)
        D = np.sqrt(D[-rank:])
        U = U[:,-rank:]
        
        sqrt_cov = U * D[None,:]
        sqrt_inv = (U / D[None,:]).T
        # original matrix is np.dot(U, U.T)

        new_A = np.dot(self.linear_part, sqrt_cov)
        new_b = self.offset - np.dot(self.linear_part, self.mean)

        mu = self.mean.copy()
        inverse_map = lambda Z: np.dot(sqrt_cov, Z) + mu[:,None]
        forward_map = lambda W: np.dot(sqrt_inv, W - mu)
        return inverse_map, forward_map, constraints(new_A, new_b)

def stack(*cons):
    """
    Combine constraints into a large constaint
    by intersection. 

    Parameters
    ----------

    cons : [`selection.affine.constraints`_]
         A sequence of constraints.

    Returns
    -------

    intersection : `selection.affine.constraints`_

    Notes
    -----

    Resulting constraint will have mean 0 and covariance $I$.

    """
    ineq, ineq_off = [], []
    eq, eq_off = [], []
    for con in cons:
        ineq.append(con.linear_part)
        ineq_off.append(con.offset)

    intersection = constraints(np.vstack(ineq), 
                               np.hstack(ineq_off))
    return intersection

def simulate_from_constraints(con, 
                              Y,
                              ndraw=1000,
                              burnin=1000,
                              white=False):
    r"""
    Use Gibbs sampler to simulate from `con`.

    Parameters
    ----------

    con : `selection.affine.constraints`_

    Y : np.float
        Point satisfying the constraint.

    ndraw : int (optional)
        Defaults to 1000.

    burnin : int (optional)
        Defaults to 1000.

    white : bool (optional)
        Is con.covariance equal to identity?

    """
    if not white:
        inverse_map, forward_map, white = con.whiten()
        Y = forward_map(Y)
    else:
        white = con
        inverse_map = lambda V: V

    white_samples = sample_truncnorm_white(white.linear_part,
                                           white.offset,
                                           Y, 
                                           ndraw=ndraw, 
                                           burnin=burnin,
                                           sigma=1.)
    return inverse_map(white_samples.T).T

def simulate_from_sphere(con, 
                         Y,
                         ndraw=1000,
                         burnin=1000,
                         white=False):
    r"""
    Use Gibbs sampler to simulate from `con` 
    intersected with (whitened) sphere of radius `np.linalg.norm(Y)`.

    Parameters
    ----------

    con : `selection.affine.constraints`_

    Y : np.float
        Point satisfying the constraint.

    ndraw : int (optional)
        Defaults to 1000.

    burnin : int (optional)
        Defaults to 1000.

    white : bool (optional)
        Is con.covariance equal to identity?

    """
    if not white:
        inverse_map, forward_map, white = con.whiten()
        Y = forward_map(Y)
    else:
        white = con
        inverse_map = lambda V: V

    white_samples = sample_truncnorm_white_sphere(white.linear_part,
                                                  white.offset,
                                                  Y, 
                                                  ndraw=ndraw, 
                                                  burnin=burnin)
    return inverse_map(white_samples.T).T

def interval_constraints(support_directions, 
                         support_offsets,
                         covariance,
                         observed_data, 
                         direction_of_interest,
                         tol = 1.e-4):
    r"""
    Given an affine in cone constraint $\{z:Az+b \leq 0\}$ (elementwise)
    specified with $A$ as `support_directions` and $b$ as
    `support_offset`, a new direction of interest $\eta$, and
    an `observed_data` is Gaussian vector $Z \sim N(\mu,\Sigma)$ 
    with `covariance` matrix $\Sigma$, this
    function returns $\eta^TZ$ as well as an interval
    bounding this value. 

    The interval constructed is such that the endpoints are 
    independent of $\eta^TZ$, hence the $p$-value
    of `Kac Rice`_
    can be used to form an exact pivot.

    Parameters
    ----------

    support_directions : np.float
         Matrix specifying constraint, $A$.

    support_offset : np.float
         Offset in constraint, $b$.

    covariance : np.float
         Covariance matrix of `observed_data`.

    observed_data : np.float
         Observations.

    direction_of_interest : np.float
         Direction in which we're interested for the
         contrast.

    tol : float
         Relative tolerance parameter for deciding 
         sign of $Az-b$.

    """

    # shorthand
    A, b, S, X, w = (support_directions,
                     support_offsets,
                     covariance,
                     observed_data,
                     direction_of_interest)

    U = np.dot(A, X) - b
    if not np.all(U  < tol * np.fabs(U).max()) and WARNINGS:
        warn('constraints not satisfied: %s' % `U`)

    Sw = np.dot(S, w)
    sigma = np.sqrt((w*Sw).sum())
    alpha = np.dot(A, Sw) / sigma**2
    V = (w*X).sum() # \eta^TZ

    # adding the zero_coords in the denominator ensures that
    # there are no divide-by-zero errors in RHS
    # these coords are never used in upper_bound or lower_bound

    zero_coords = alpha == 0
    RHS = (-U + V * alpha) / (alpha + zero_coords)
    RHS[zero_coords] = np.nan

    pos_coords = alpha > tol * np.fabs(alpha).max()
    if np.any(pos_coords):
        upper_bound = RHS[pos_coords].min()
    else:
        upper_bound = np.inf
    neg_coords = alpha < -tol * np.fabs(alpha).max()
    if np.any(neg_coords):
        lower_bound = RHS[neg_coords].max()
    else:
        lower_bound = -np.inf

    return lower_bound, V, upper_bound, sigma

def selection_interval(support_directions, 
                       support_offsets,
                       covariance,
                       observed_data, 
                       direction_of_interest,
                       tol = 1.e-4,
                       alpha = 0.05,
                       UMAU=True):
    """
    Given an affine in cone constraint $\{z:Az+b \leq 0\}$ (elementwise)
    specified with $A$ as `support_directions` and $b$ as
    `support_offset`, a new direction of interest $\eta$, and
    an `observed_data` is Gaussian vector $Z \sim N(\mu,\Sigma)$ 
    with `covariance` matrix $\Sigma$, this
    function returns a confidence interval
    for $\eta^T\mu$.

    Parameters
    ----------

    support_directions : np.float
         Matrix specifying constraint, $A$.

    support_offset : np.float
         Offset in constraint, $b$.

    covariance : np.float
         Covariance matrix of `observed_data`.

    observed_data : np.float
         Observations.

    direction_of_interest : np.float
         Direction in which we're interested for the
         contrast.

    tol : float
         Relative tolerance parameter for deciding 
         sign of $Az-b$.

    UMAU : bool
         Use the UMAU interval, or two-sided pivot.

    Returns
    -------

    selection_interval : (float, float)

    """

    lower_bound, V, upper_bound, sigma = interval_constraints( \
        support_directions, 
        support_offsets,
        covariance,
        observed_data, 
        direction_of_interest,
        tol=tol)

    truncated = truncated_gaussian([(lower_bound, upper_bound)], sigma=sigma)
    if UMAU:
        _selection_interval = truncated.UMAU_interval(V, alpha)
    else:
        _selection_interval = truncated.naive_interval(V, alpha)
    
    return _selection_interval

def gibbs_test(affine_con, Y, direction_of_interest,
               ndraw=5000,
               burnin=2000,
               white=False,
               alternative='two-sided',
               sigma_known=False):
    """
    A Monte Carlo significance test for
    a given function of `con.mean`.

    Parameters
    ----------

    affine_con : `selection.affine.constraints`_

    Y : np.float
        Point satisfying the constraint.

    direction_of_interest: np.float
        Which linear function of `con.mean` is of interest?
        (a.k.a. $\eta$ in many of related papers)

    ndraw : int (optional)
        Defaults to 1000.

    burnin : int (optional)
        Defaults to 1000.

    white : bool (optional)
        Is con.covariance equal to identity?

    alternative : str
        One of ['greater', 'less', 'two-sided']

    """
    eta = direction_of_interest # shorthand

    if alternative not in ['greater', 'less', 'two-sided']:
        raise ValueError("expecting alternative to be in ['greater', 'less', 'two-sided']")

    if not sigma_known:
        Z = simulate_from_sphere(affine_con,
                                 Y,
                                 ndraw=ndraw,
                                 burnin=burnin,
                                 white=white)
    else:
        Z = simulate_from_constraints(affine_con,
                                      Y,
                                      ndraw=ndraw,
                                      burnin=burnin,
                                      white=white)
        
    null_statistics = np.dot(Z, eta)
    observed = (eta*Y).sum()
    if alternative == 'greater':
        pvalue = (null_statistics >= observed).mean()
    elif alternative == 'less':
        pvalue = (null_statistics <= observed).mean()
    else:
        pvalue = (np.fabs(null_statistics) <= np.fabs(observed)).mean()
    return pvalue, Z