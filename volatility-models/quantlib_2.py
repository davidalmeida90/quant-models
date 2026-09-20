"""extra, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

# pip install tf-quant-finance tensorflow tf-keras QuantLib
import tensorflow as tf
from tf_quant_finance.math import pde


@tf.function
def american_option(number_grid_points, time_delta, strike, volatility,
                    risk_free_rate, expiry, dtype=tf.float64):
    s_min, s_max = 0.01, 300.
    grid = pde.grids.uniform_grid(minimums=[s_min], maximums=[s_max],
                                  sizes=[number_grid_points], dtype=dtype)
    final_values_grid = tf.nn.relu(grid[0] - strike)

    def second_order_coeff_fn(t, grid): return [[volatility**2 * grid[0]**2 / 2]]
    def first_order_coeff_fn(t, grid):  return [risk_free_rate * grid[0]]
    def zeroth_order_coeff_fn(t, grid): return -risk_free_rate

    # this is what makes it AMERICAN: after every backward step, floor the value at
    # the intrinsic payoff. that single line is the early exercise decision.
    def values_transform_fn(t, grid, values):
        return grid, tf.maximum(values, tf.nn.relu(grid[0] - strike))

    v, g, _, _ = pde.fd_solvers.solve_backward(
        start_time=expiry, end_time=0, coord_grid=grid,
        values_grid=final_values_grid, time_step=time_delta,
        values_transform_fn=values_transform_fn, ...)
    return v, g[0]

# strike, volatility and risk_free_rate are all (n, 1) tensors.
# ONE call prices n contracts. that is the entire trick.
