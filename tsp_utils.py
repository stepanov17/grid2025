import json
import math
import numpy as np
from scipy.optimize import fsolve, minimize


P_MIN = 0.01
P_MAX = 20.0


def tsp_inv_cdf(y: np.ndarray, a: float, m: float, b: float, p: float) -> np.ndarray:
    """inverse CDF for TSP(a, m, b, p) distribution; q = (m - a) / (b - a)"""

    if b < a:
        raise ValueError(f"b ({b}) must be >= a ({a})")
    if b == a:
        return np.full_like(y, a)

    q_val = (m - a) / (b - a)
    res = np.zeros_like(y)
    mask = y <= q_val

    if np.any(mask):  # left part of the distribution (y <= q_val)
        if m == a:
            res[mask] = a
        else:
            term1 = y[mask] * np.power(m - a, p - 1.) * (b - a)
            res[mask] = a + np.power(term1, 1. / p)

    if np.any(~mask):  # right part of the distribution (y > q_val)
        if m == b:
            res[~mask] = b
        else:
            term2 = (1. - y[~mask]) * np.power(b - m, p - 1.) * (b - a)
            res[~mask] = b - np.power(term2, 1. / p)

    return res


def _equations_fixed_boundaries(variables: list[float],
                                x0: float,
                                u: float,
                                a: float,
                                b: float) -> list:
    """prepare a list of equations for "fixed boundaries", case: (x0, u) -> (m, p)"""
    m, p = variables
    # equation (1): x0
    eq1 = (a + (p - 1) * m + b) / (p + 1) - x0
    # equation (2): u
    term = (m - a) * (b - m) / ((b - a)**2)
    numerator = p - 2. * (p - 1) * term
    denominator = (p + 1)**2 * (p + 2)
    eq2 = (b - a)**2 * numerator / denominator - u**2
    return [eq1, eq2]


def _equations_coverage_interval(variables: list[float],
                                 x0: float,
                                 u: float,
                                 c1: float,
                                 c2: float,
                                 p0: float) -> list:
    """prepare a list of equations for "coverage interval boundaries", case: (x0, u, c1, c2, p0) -> (a, m, b, p)"""
    a, m, b, p = variables
    # equation (1): x0
    eq1 = (a + (p - 1) * m + b) / (p + 1) - x0
    # equation (2): u
    term = (m - a) * (b - m) / ((b - a)**2)
    numerator = p - 2. * (p - 1) * term
    denominator = (p + 1)**2 * (p + 2)
    eq2 = (b - a)**2 * numerator / denominator - u**2
    # equation (3): c1
    eq3 = a + (m - a) * math.pow(
        0.5 * (b - a) / (m - a) * (1 - p0), 1. / p) - c1
    # equation (4): c2
    eq4 = b - (b - m) * math.pow(
        0.5 * (b - a) / (b - m) * (1 - p0), 1. / p) - c2
    return [eq1, eq2, eq3, eq4]


def get_tsp_params_by_mean_u(
        x0: float, u: float, a: float, b: float) -> dict[str, float]:
    """
    try to get a, b, m, p parameters of TSP(a, b, m, p) basing on its mean x0 and uncertainty (standard deviation) u
    """
    initial_approximation = np.array([0.5 * (a + b), 2.], dtype=np.float64)
    values = fsolve(_equations_fixed_boundaries,
                    initial_approximation,
                    args=(x0, u, a, b),
                    xtol=1.e-5)

    res = dict(zip(["m", "p"], values.tolist()))

    if not all([res["p"] > 0, a <= res["m"] <= b]):
        raise ArithmeticError("cannot fit TSP parameters")

    res["a"] = a
    res["b"] = b

    res_reordered = {k: res[k] for k in ["a", "m", "b", "p"]}
    return res_reordered


def get_tsp_params_by_cov_interval_mean_u(x0: float,
                                          u: float,
                                          c1: float,
                                          c2: float,
                                          p0: float) -> dict[str, float]:
    """
    try to get a, b, m, p parameters of TSP(a, b, m, p)
    basing on its mean x0, uncertainty (standard deviation) u
    and coverage interval (c1, c2) (corresponding to the probability level of p0)
    """
    initial_approximation = np.array(
        [c1, 0.5 * (c1 + c2), c2, 2.], dtype=np.float64)
    values = fsolve(_equations_coverage_interval,
                    initial_approximation,
                    args=(x0, u, c1, c2, p0),
                    xtol=1.e-5)

    res = dict(zip(["a", "m", "b", "p"], values.tolist()))

    if not all([res["p"] > 0, res["a"] <= res["m"] <= res["b"]]):
        raise ArithmeticError("cannot fit TSP parameters")

    if (res["b"] - res["a"]) / (c2 - c1) > 1e2:  # the range is too wide
        raise ArithmeticError("cannot fit TSP parameters")

    return res


def _fit_tsp_objective(params, sorted_sample, n):
    """
    a functional to minimize with transformed parameters to ensure a < m < b (inverse mapping method)
    """
    a, m_transformed, b, p = params
    # transform m_transformed back to ensure a < m < b
    m = a + (b - a) * 1 / (1 + np.exp(-m_transformed))
    u = (np.arange(1, n + 1) - 0.5) / n  # vector of (i - 0.5) / n
    inv_cdf_vals = tsp_inv_cdf(u, a, m, b, p)
    return np.sum((inv_cdf_vals - sorted_sample) ** 2)


def get_tsp_params_by_sample_data(sample: np.array) -> dict:
    """
    estimate TSP(a, m, b, p) parameters using an inverse mapping method
    """

    n = len(sample)
    sorted_sample = np.sort(sample)

    # initial guess
    a_init = np.min(sample) - 0.1 * np.std(sample)
    b_init = np.max(sample) + 0.1 * np.std(sample)
    m_init = np.median(sample)
    p_init = 2.0

    # transform initial m to unconstrained space
    m_transformed_init = np.log((m_init - a_init) / (b_init - m_init))

    bounds = [
        (None, None),   # a
        (None, None),   # m_transformed (unconstrained)
        (None, None),   # b
        (P_MIN, P_MAX)  # p > 0
    ]

    result = minimize(
        fun=_fit_tsp_objective,
        x0=[a_init, m_transformed_init, b_init, p_init],
        args=(sorted_sample, n),
        bounds=bounds,
        method="L-BFGS-B"
    )

    # extract and transform parameters back
    a_est, m_transformed_est, b_est, p_est = result.x
    m_est = (a_est + (b_est - a_est) * 1 / (1 + np.exp(-m_transformed_est)))  # sigmoid
    return {"a": a_est, "m": m_est, "b": b_est, "p": p_est}


def _fit_tsp_objective_histogram(params, bin_edges, cumulative_freq, total_points):
    """
    objective function for histogram data
    """
    a, m_transformed, b, p = params
    # transform m_transformed back to ensure a < m < b
    m = a + (b - a) * 1 / (1 + np.exp(-m_transformed))

    # calculate theoretical quantiles for the cumulative frequencies
    inv_cdf_vals = tsp_inv_cdf(cumulative_freq, a, m, b, p)

    # Use bin edges as empirical quantiles
    return np.sum((inv_cdf_vals - bin_edges) ** 2)


def get_tsp_params_by_hist_data(bin_centers: np.array, frequencies: np.array) -> dict:
    """
    try to get a, b, m, p parameters of TSP(a, b, m, p) basing on a histogram data
    """

    total_points = np.sum(frequencies)
    cumulative_freq = (np.cumsum(frequencies) - 0.5 * frequencies) / total_points

    # initial guess
    min_val, max_val = np.min(bin_centers), np.max(bin_centers)
    a_init = min_val - 0.1 * (max_val - min_val)
    b_init = max_val + 0.1 * (max_val - min_val)
    m_init = np.median(bin_centers)
    p_init = 2.0

    m_transformed_init = np.log((m_init - a_init) / (b_init - m_init))

    bounds = [
        (None, None),   # a
        (None, None),   # m_transformed
        (None, None),   # b
        (P_MIN, P_MAX)  # p > 0
    ]

    def objective_func(params):
        a, m_transformed, b, p = params
        m = a + (b - a) * 1 / (1 + np.exp(-m_transformed))
        inv_cdf_vals = tsp_inv_cdf(cumulative_freq, a, m, b, p)
        return np.sum((inv_cdf_vals - bin_centers) ** 2)

    result = minimize(
        objective_func,
        x0=[a_init, m_transformed_init, b_init, p_init],
        bounds=bounds,
        method="L-BFGS-B"
    )

    a_est, m_transformed_est, b_est, p_est = result.x
    m_est = a_est + (b_est - a_est) * 1 / (1 + np.exp(-m_transformed_est))

    return {"a": a_est, "m": m_est, "b": b_est, "p": p_est}


def process_inputs(annotation: dict) -> dict:
    """
    process annotation dictionary by:
    1. converting TSP parameters to canonical form
    2. expanding comma-separated keys into individual keys
    """
    result = annotation.copy()

    if "inputs" not in result:
        raise ValueError("no inputs provided")

    inputs = result["inputs"].copy()

    # validate variable name
    def is_valid_variable_name(name: str) -> bool:
        """check if string is a valid Python variable name"""
        if not name:
            return False
        if not (name[0].isalpha() or name[0] == "_"):
            return False
        for char in name[1:]:
            if not (char.isalnum() or char == "_"):
                return False
        return True

    # first pass: validate and convert parameters
    for key, params in inputs.items():
        if not isinstance(params, dict) or "type" not in params:
            raise ValueError(f"'{key}': invalid description for input '{key}'")

        # validate normal distribution
        if params["type"] == "normal":
            required_keys = {"type", "mu", "u"}
            if not required_keys.issubset(params.keys()):
                raise ValueError(f"'{key}': normal distribution requires keys: {required_keys}")
            if params["u"] <= 0:
                raise ValueError(f"'{key}': parameter 'u' must be positive")

        # validate and convert TSP distributions
        elif params["type"] == "tsp":
            keys_set = set(params.keys())
            converted = {"type": "tsp"}

            # a) canonical TSP parameters
            if keys_set == {"type", "a", "m", "b", "p"}:
                if params["p"] <= 0:
                    raise ValueError(f"'{key}': parameter 'p' must be positive for TSP distribution")
                if not (params["a"] <= params["m"] <= params["b"]):
                    raise ValueError(f"'{key}': TSP parameters must satisfy: a <= m <= b")
                if params["a"] >= params["b"]:
                    raise ValueError(f"'{key}': TSP parameters must satisfy: a < b")

            # b) mu, u, a, b
            elif keys_set == {"type", "mu", "u", "a", "b"}:
                if params["u"] <= 0:
                    raise ValueError(f"'{key}': parameter 'u' must be positive")
                if not (params["a"] < params["mu"] < params["b"]):
                    raise ValueError(f"'{key}': TSP parameters must satisfy: a < mu < b")

                converted.update(get_tsp_params_by_mean_u(params["mu"], params["u"], params["a"], params["b"]))
                inputs[key] = converted

            # c) mu, u, c1, c2, cov_prob
            elif keys_set == {"type", "mu", "u", "c1", "c2", "cov_prob"}:
                if params["u"] <= 0:
                    raise ValueError(f"'{key}': parameter 'u' must be positive for TSP distribution")
                if not (params["c1"] < params["mu"] < params["c2"]):
                    raise ValueError(f"'{key}': TSP parameters must satisfy: c1 < mu < c2")
                if not (0.75 <= params["cov_prob"] < 1):
                    raise ValueError(f"'{key}': parameter 'cov_prob' must be in range [0.75, 1)")

                converted.update(get_tsp_params_by_cov_interval_mean_u(
                    params["mu"], params["u"], params["c1"], params["c2"], params["cov_prob"]))
                inputs[key] = converted

            # d) mu, u, cov_factor, cov_prob
            elif keys_set == {"type", "mu", "u", "cov_factor", "cov_prob"}:
                if params["u"] <= 0:
                    raise ValueError(f"'{key}': Parameter 'u' must be positive for TSP distribution")
                if params["cov_factor"] <= 0:
                    raise ValueError(f"'{key}': non-positive coverage factor")
                if not (0.75 <= params["cov_prob"] < 1):
                    raise ValueError(f"'{key}': Parameter 'cov_prob' must be in range [0.75, 1)")

                mu, u, cov_factor = params["mu"], params["u"], params["cov_factor"]

                converted.update(get_tsp_params_by_cov_interval_mean_u(
                    mu, u, mu - cov_factor * u, mu + cov_factor * u, params["cov_prob"]))
                inputs[key] = converted

            # e) histogram
            elif keys_set == {"type", "histogram_path"}:
                # if not isinstance(params["histogram_path"], str):
                #     raise ValueError("Parameter 'histogram_path' must be a string")

                bins, freqs = np.loadtxt(params["histogram_path"], usecols=(0, 1), unpack=True)
                converted.update(get_tsp_params_by_hist_data(bins, freqs))
                inputs[key] = converted

            # f) sample data
            elif keys_set == {"type", "sample_path"}:
                # if not isinstance(params["sample_path"], str):
                #     raise ValueError("Parameter 'histogram_path' must be a string")

                sample = np.loadtxt(params["sample_path"], dtype=float)
                converted.update(get_tsp_params_by_sample_data(sample))
                inputs[key] = converted

            else:
                raise ValueError(f"invalid parameter set for TSP distribution: {keys_set}")

    # second pass: expand composite keys and validate variable names
    expanded_inputs = {}
    keys_to_remove = []

    for key, params in inputs.items():
        # check if key is a composite key (contains commas or whitespace-separated variables)
        if isinstance(key, str):
            if "," in key:
                variable_names = [name.strip() for name in key.split(",") if name.strip()]
            else:
                variable_names = key.split()

            if len(variable_names) > 1:  # it's a composite key
                for var_name in variable_names:
                    if not is_valid_variable_name(var_name):
                        raise ValueError(f"invalid variable name: '{var_name}'")
                    if var_name in expanded_inputs:
                        raise ValueError(f"duplicate variable name: {var_name}")
                    expanded_inputs[var_name] = params.copy()

                # mark the composite key for removal
                keys_to_remove.append(key)
                continue

        if isinstance(key, str) and not is_valid_variable_name(key):  # validate non-composite key names
            raise ValueError(f"invalid variable name: '{key}'")

        if key in expanded_inputs:
            raise ValueError(f"duplicate variable name: {key}")
        expanded_inputs[key] = params

    # remove composite keys and update inputs
    for key in keys_to_remove:
        if key in expanded_inputs:
            del expanded_inputs[key]

    result["inputs"] = expanded_inputs
    return result


# if __name__ == "__main__":  # for debug purposes
# 
#     test_annotation = {
#         "inputs": {
#             "x_sample": {"type": "tsp", "sample_path": "../example/test_sample.txt"},
#             "x_hist": {"type": "tsp", "histogram_path": "../example/test_hist.txt"},
#             "x_m_u, x_m_u_2": {"type": "tsp", "a": 0.0, "b": 1.0, "mu": 0.6, "u": 0.16},
#             "x_m_u_U": {"type": "tsp", "mu": 0.6, "u": 0.164, "c1": 0.23, "c2": 0.87, "cov_prob": 0.95},
#             "x_m_u_k": {"type": "tsp", "mu": 0.5, "u": 0.164, "cov_factor": 1.99, "cov_prob": 0.95},
#         },
#     }
#     print(json.dumps(process_inputs(test_annotation), indent=4))
