from __future__ import division, print_function

import numpy as np

#np.seterr(divide='ignore', invalid='ignore')
from scipy.stats import norm
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from selection.randomized.selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C

def approx_reference(grid,
                     observed_target,
                     cov_target,
                     cov_target_score,
                     init_soln,
                     cond_mean,
                     cond_cov,
                     logdens_linear,
                     linear_part,
                     offset,
                     solve_args={'tol': 1.e-15}
                     ):

    if np.asarray(observed_target).shape in [(), (0,)]:
        raise ValueError('no target specified')

    observed_target = np.atleast_1d(observed_target)
    prec_target = np.linalg.inv(cov_target)
    target_lin = - logdens_linear.dot(cov_target_score.T.dot(prec_target))

    prec_opt = np.linalg.inv(cond_cov)

    ref_hat =[]
    solver = solve_barrier_affine_C
    for k in range(grid.shape[0]):
        cond_mean_grid = target_lin.dot(np.asarray([grid[k]])) + (cond_mean - target_lin.dot(observed_target))
        conjugate_arg = prec_opt.dot(cond_mean_grid)

        val, _, _ = solver(conjugate_arg,
                           prec_opt,
                           init_soln,
                           linear_part,
                           offset,
                           **solve_args)

        ref_hat.append(-val-(conjugate_arg.T.dot(cond_cov).dot(conjugate_arg)/2.))

    return np.asarray(ref_hat)



def approx_reference_adaptive(grid,
                              observed_target,
                              cov_target,
                              cov_target_score,
                              init_soln,
                              cond_mean,
                              cond_cov,
                              logdens_linear,
                              linear_part,
                              offset,
                              subgrad,
                              weights,
                              score,
                              nonzero,
                              scaling,
                              solve_args={'tol': 1.e-15}
                              ):

    if np.asarray(observed_target).shape in [(), (0,)]:
        raise ValueError('no target specified')

    observed_target = np.atleast_1d(observed_target)
    prec_target = np.linalg.inv(cov_target)

    linear_coef = cov_target_score.T.dot(prec_target)
    linear_offset = score - linear_coef.dot(observed_target)

    target_lin = - logdens_linear.dot(linear_coef)
    N_sel = cond_mean  - target_lin.dot(observed_target) + logdens_linear.dot(subgrad)

    prec_opt = np.linalg.inv(cond_cov)

    ref_hat = []
    solver = solve_barrier_affine_C
    for k in range(grid.shape[0]):

        weight_grid = 1./np.abs(scaling.dot(linear_coef.dot(np.asarray([grid[k]])) + linear_offset))
        cond_mean_grid = target_lin.dot(np.asarray([grid[k]])) + N_sel - \
                         logdens_linear.dot(np.diag(weight_grid)).dot(np.true_divide(subgrad, weights))
        conjugate_arg = prec_opt.dot(cond_mean_grid)
        jacob = np.prod(weight_grid[~nonzero])

        val, _, _ = solver(conjugate_arg,
                           prec_opt,
                           init_soln,
                           linear_part,
                           offset,
                           **solve_args)

        ref_hat.append(-val - (conjugate_arg.T.dot(cond_cov).dot(conjugate_arg))/ 2. + np.log(jacob))

    return np.asarray(ref_hat)

def approx_density(grid,
                   mean_parameter,
                   cov_target,
                   approx_log_ref,
                   lee_lower=float('inf'),
                   lee_upper=float('inf')):
    _approx_density = []
    _approx_naive_density = []
    _approx_density_lee = []
    for k in range(grid.shape[0]):
        _approx_density.append(
            np.exp(-np.true_divide((grid[k] - mean_parameter) ** 2, 2 * cov_target) + approx_log_ref[k]))
        _approx_naive_density.append(
            norm.pdf((grid[k] - mean_parameter) / np.sqrt(cov_target)))

    if lee_lower == float('inf'):
        grid_lower = 0
    else:
        grid_lower = np.argmin(np.abs(grid - lee_lower))
    if lee_upper == float('inf'):
        grid_upper = grid.shape[0]
    else:
        grid_upper = np.argmin(np.abs(grid - lee_upper))
    for k in range(grid.shape[0])[grid_lower:grid_upper]:
        _approx_density_lee.append(-np.true_divide((grid[k] - mean_parameter) ** 2, 2 * cov_target))

    if np.asarray(_approx_density_lee).sum() == 0:
        _approx_density_ = _approx_density
    else:
        _approx_density_ = np.asarray(_approx_density) / (np.asarray(_approx_density).sum())
    if np.asarray(_approx_naive_density).sum() == 0:
        _approx_naive_density_ = _approx_naive_density
    else:
        _approx_naive_density_ = np.asarray(_approx_naive_density) / (np.asarray(_approx_naive_density).sum())
    if np.asarray(_approx_density_lee).sum() == 0:
        _approx_density_lee_ = _approx_density_lee
    else:
        _approx_density_lee_ = np.asarray(_approx_density_lee) / (np.asarray(_approx_density_lee).sum())
    return np.cumsum(_approx_density_), np.cumsum(_approx_naive_density_), np.cumsum(_approx_density_lee_)

def approx_ci(param_grid,
              grid,
              cov_target,
              approx_log_ref,
              indx_obsv=100,
              lee_lower=-float('inf'),
              lee_upper=float('inf')):

    area = np.zeros(param_grid.shape[0])
    #area_naive = np.zeros(param_grid.shape[0])
    #area_lee = np.zeros(param_grid.shape[0])

    for k in range(param_grid.shape[0]):
        area_vec, area_vec_naive, area_vec_lee = approx_density(grid,
                                                                param_grid[k],
                                                                cov_target,
                                                                approx_log_ref,
                                                                lee_lower,
                                                                lee_upper)
        area[k] = area_vec[indx_obsv]
        #area_naive[k] = area_vec_naive[indx_obsv]
        #area_lee[k] = area_vec_lee[indx_obsv]

    region = param_grid[(area > 0.05) & (area < 0.95)]
    #region_naive = param_grid[(area_naive >= 0.05) & (area_naive <= 0.95)]
    #region_lee = param_grid[(area_lee >= 0.05) & (area_lee <= 0.95)]
    if region.size > 0:
        return np.nanmin(region), np.nanmax(region)
    else:
        return 0., 0.
    #if region_naive.size > 0:
    #    return np.nanmin(region_naive), np.nanmax(region_naive)
    #else:
        return 0., 0.

