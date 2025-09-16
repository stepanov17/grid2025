import datetime
import logging
import math
import json
import os
import re
import sys

import numpy as np
import numexpr as ne
from scipy.stats import norm
import matplotlib.pyplot as plt

from tsp_utils import process_inputs, tsp_inv_cdf


logging.basicConfig(
    format="%(asctime)s %(levelname)-9s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


def generate_samples(inputs: dict, n: int, correlations: list[list[float]] = None) -> dict:
    """generate n correlated samples and return variable name to values dict"""

    keys = list(inputs.keys())
    num_vars = len(keys)

    if correlations is None:
        corr_matrix = np.eye(num_vars)
    else:
        corr_matrix = np.array(correlations)
        if corr_matrix.shape != (num_vars, num_vars):
            raise ValueError(f"correlation matrix must be of shape ({num_vars}, {num_vars})")

    means = np.zeros(num_vars)
    try:
        chol = np.linalg.cholesky(corr_matrix)  # try to use Cholesky decomposition
        independent_normals = np.random.normal(size=(n, num_vars))
        correlated_normals = independent_normals @ chol.T
    except np.linalg.LinAlgError:
        correlated_normals = np.random.multivariate_normal(means, corr_matrix, n)

    uniform_data = norm.cdf(correlated_normals)

    result = {}
    for i, key in enumerate(keys):
        params = inputs[key]
        dist_type = params["type"]

        if dist_type == "tsp":
            a, m_val, b, p_val = params["a"], params["m"], params["b"], params["p"]
            sample = tsp_inv_cdf(uniform_data[:, i], a, m_val, b, p_val)
            result[key] = sample

        elif dist_type == "normal":
            loc, scale = params["mu"], params["u"]
            sample = norm.ppf(uniform_data[:, i], loc=loc, scale=scale)
            result[key] = sample

        else:
            raise ValueError(f"unsupported distribution type {dist_type} for {key}")

    return result


def check_correlation_matrix(matrix: list, expected_dim: int) -> bool:
    """check if correlation matrix is valid"""
    try:
        m = np.array(matrix, dtype=float)
    except:
        return False

    # 1. is square
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False

    # 2. check dimension
    n = m.shape[0]
    if n != expected_dim:
        return False

    # 3. check diag
    if not np.allclose(np.diag(m), np.ones(n)):
        return False

    # 4. check symmetry
    if not np.allclose(m, m.T):
        return False

    # 5. check elements range
    if np.any(m < -1.0) or np.any(m > 1.0):
        return False

    # 6. check if positive semi-definite matrix
    try:
        eigenvalues = np.linalg.eigvalsh(m)
        if np.any(eigenvalues < -1e-8):
            return False
    except:
        return False

    return True


def get_shortest_coverage_interval(sorted_data: np.ndarray, p0: float) -> list:
    """get the shortest coverage interval for the probability level of p0"""

    n1 = round(len(sorted_data) * p0)
    n2 = len(sorted_data) - n1

    l0 = float("inf")
    a, b = None, None

    for start in range(n2 + 1):

        c1, c2 = sorted_data[start], sorted_data[start + n1 - 1]
        l = c2 - c1
        if l < l0:
            a, b = c1, c2

    return [a, b]


def add_output_stats(data: np.ndarray, output: dict) -> None:
    """append stats for the transformed distribution"""

    n_digs = output["n_digs"]

    funcs = {"min": np.min, "max": np.max, "mean": np.mean, "median": np.median, "stdev": np.std}
    results = {k: round(f(data), n_digs) for k, f in funcs.items()}

    title = "y: " + ", ".join(
        [f"{key} = {value:.{n_digs}f}" for key, value in results.items()])
    plt.title(title)

    # calculate coverage interval
    data.sort()
    n = len(data)

    cov_interval_types = (
        "probabilistically_symmetric", "left", "right", "shortest")
    coverage_interval_type = results.get(
        "coverage_interval", "probabilistically_symmetric")

    coverage_probability = results.get("coverage_probability", 0.95)
    if not 0.5 <= coverage_probability < 1:
        raise ValueError(
            f"invalid coverage probability value: {coverage_probability}; "
            "expecting to be in the range [0.5, 1)")

    if coverage_interval_type == "probabilistically_symmetric":
        dp = 0.5 * (1. - coverage_probability)
        coverage_interval = [data[round(n * dp)], data[round(n * (1 - dp))]]
    elif coverage_interval_type == "left":
        coverage_interval = [data[0], data[round(n * coverage_probability)]]
    elif coverage_interval_type == "right":
        coverage_interval = [data[round(n * (1 - coverage_probability))], data[-1]]
    elif coverage_interval_type == "shortest":
        coverage_interval = get_shortest_coverage_interval(data, coverage_probability)
    else:
        raise ValueError(
            f"invalid coverage interval type: {coverage_interval_type}; "
            f"expected to be one of {cov_interval_types}")

    results["coverage_interval"] = [round(coverage_interval[0], n_digs), round(coverage_interval[1], n_digs)]
    results["coverage_interval_length"] = round(coverage_interval[1] - coverage_interval[0], n_digs)

    output["results"] = results


def compute(inputs: dict[str, np.ndarray], model: str, subexpressions: list[str] = None) -> np.ndarray:
    """compute model values for all simulations with hybrid math/numexpr support"""

    n_sim = len(next(iter(inputs.values())))
    for name, arr in inputs.items():
        if len(arr) != n_sim:
            raise ValueError(f"array length for '{name}' ({len(arr)}) doesn't match expected ({n_sim})")

    local_vars = {}  # a local namespace for the computations
    elementwise_funcs = [  # add numpy functions (default - for array operations)
        "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
        "exp", "log", "log10", "sqrt", "abs", "absolute", "floor", "ceil", "round", "sign", "isfinite", "isinf", "isnan"
    ]
    for func_name in elementwise_funcs:
        if hasattr(np, func_name):
            local_vars[func_name] = getattr(np, func_name)

    # add special handling for min/max
    local_vars["min"] = np.minimum
    local_vars["max"] = np.maximum
    # add constants
    local_vars["pi"] = np.pi
    local_vars["e"] = np.e
    local_vars["inf"] = np.inf
    local_vars["nan"] = np.nan

    def process_math_functions(expression, is_subexpression=False) -> tuple:
        """
        process math.func() calls by evaluating them separately and storing results
        (for subexpressions, we can compute them directly; for main model, we need to create intermediate variables)
        """
        # find all math.function() patterns
        math_pattern = r"math\.(\w+)\(([^)]+)\)"
        matches = re.findall(math_pattern, expression)

        if not matches:
            return expression, {}

        math_results = {}
        processed_expression = expression

        for i, (math_func, args) in enumerate(matches):
            if hasattr(math, math_func):
                # create a unique variable name for this math function result
                var_name = f"math_{math_func}_{i}"

                # for subexpressions, we can compute directly
                if is_subexpression:
                    # evaluate the arguments first
                    try:
                        arg_values = ne.evaluate(args, local_dict={**local_vars, **inputs})

                        # apply math function element-wise
                        if isinstance(arg_values, np.ndarray):
                            # vectorize the math function for arrays
                            result = np.array([getattr(math, math_func)(x) for x in arg_values])
                        else:
                            # scalar case
                            result = getattr(math, math_func)(arg_values)

                        math_results[var_name] = result
                        processed_expression = processed_expression.replace(f"math.{math_func}({args})", var_name)
                    except Exception as e:
                        raise ValueError(f"error evaluating math.{math_func}({args})") from e
                else:
                    # for main model, just replace with variable reference
                    processed_expression = processed_expression.replace(f"math.{math_func}({args})", var_name)

        return processed_expression, math_results

    # compute subexpressions if provided
    math_intermediate_vars = {}

    if subexpressions:
        for i, expr in enumerate(subexpressions):  # calculate each subexpression
            # process math functions in subexpression
            processed_expr, math_vars = process_math_functions(expr, is_subexpression=True)
            math_intermediate_vars.update(math_vars)

            # replace variables with their values
            expr_vars = {}
            for var_name, var_values in inputs.items():
                expr_vars[var_name] = var_values

            # add functions and math results to evaluation
            expr_vars.update(local_vars)
            expr_vars.update(math_vars)

            # compute subexpression
            try:
                result = ne.evaluate(processed_expr, local_dict=expr_vars)
                local_vars[f"sub_{i}"] = result
            except Exception as e:
                raise ValueError(f"error computing subexpression {i}: {expr}") from e

        for i in range(len(subexpressions)):  # replace subexpressions[i] notation
            model = model.replace(f"subexpressions[{i}]", f"sub_{i}")

    # process math functions in the final model
    processed_model, final_math_vars = process_math_functions(model, is_subexpression=True)
    math_intermediate_vars.update(final_math_vars)

    # add input variables, functions, and math results
    local_vars.update(inputs)
    local_vars.update(math_intermediate_vars)
    try:
        result = ne.evaluate(processed_model, local_dict=local_vars)  # evaluate
        return result
    except Exception as e:
        raise ValueError(f"error computing model: {processed_model}") from e


def transform(annotation: dict) -> np.ndarray:
    """transform the input distributions"""

    annotation = process_inputs(annotation)
    n_trials = int(annotation.get("n_trials", 1000000))  # may be float, e.g. 2e6
    min_n_trials = 10000
    if n_trials < min_n_trials:
        log.warning(f"n(trials) is too low, setting to {min_n_trials}")
        n_trials = min_n_trials

    inputs = annotation["inputs"]

    correlations = annotation.get("correlations")  # optional
    if correlations is not None:
        if not check_correlation_matrix(correlations, len(inputs)):
            raise ValueError(f"invalid correlation matrix: {correlations}")
    samples = generate_samples(inputs, n_trials, correlations)

    res = compute(samples, annotation["transform"], annotation.get("subexpressions"))
    return res


def add_histogram(data: np.ndarray, n_digs: int, n_bins: int, output: dict) -> None:
    """histogram: plot and add to output json if required"""

    freqs, bins, _ = plt.hist(data, bins=n_bins, color="lightseagreen")
    # draw coverage interval and mean
    coverage_interval = output.get("results", {}).get("coverage_interval", [])
    if coverage_interval:
        plt.plot(coverage_interval, [0., 0.], linewidth=7., color="red")
    out_mean = output.get("results", {}).get("mean")
    if out_mean is not None:
        max_y = plt.gca().get_ylim()[1]
        plt.plot([out_mean, out_mean], [0., 0.75 * max_y], linewidth=3., color="red", linestyle="dotted")
    plt.grid()

    hist_data = {}
    for (val, freq) in zip(bins, freqs):
        hist_data[round(val, n_digs)] = int(freq)

    bin_fmt = f"%.{n_digs}f"
    if output.get("add_histogram", True):
        output.setdefault("results", {})["histogram"] = (
            dict(zip([bin_fmt % round(b, n_digs) for b in bins],
                     [int(f) for f in freqs])))


def main(input_file: str) -> None:

    try:
        with open(input_file, "r") as f:
            annotation = json.load(f)
        log.info("start calculations")

        transformed_data = transform(annotation)
        out = annotation.copy()
        add_output_stats(transformed_data, out)
        add_histogram(transformed_data, annotation.get("n_digs", 3),
                      annotation.get("n_histogram_bins", 100), out)

        file_name, file_extension = os.path.splitext(input_file)
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        out_json = f"{file_name}_results_{suffix}{file_extension}"
        out_png = f"{file_name}_results_{suffix}.png"
        log.info(f"done, the output will be saved as: {out_json}, {out_png}")
        with open(out_json, "w") as f:
            json.dump(out, f, indent=4)
        plt.gcf().set_size_inches(12, 9)
        plt.savefig(out_png, dpi=200)

    except Exception as e:
        log.error(f"{type(e).__name__}: {e}", exc_info=True)


if __name__ == "__main__":

    args = sys.argv
    if len(args) < 2:
        log.error("no input JSON file provided; usage: python mc.py input.json")
        exit(1)
    main(args[1])


# if __name__ == "__main__":  # for debug purposes
#     main("./test_input.json")
