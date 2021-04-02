import numpy as np
from scipy import optimize


def equally_weighted(covariance_matrix, verbose=0):
    """
    This function computes the weights of the Equal-Weighted strategy.
    :param n*n ndarray covariance_matrix: covariance matrix.
    :param int verbose: Doesn't do anything.
        This argument is passed to provide the same
        signature for all risk-based portfolio strategies.
    :return: vector of weights.
    :rtype: n*1 ndarray
    """

    n = covariance_matrix.shape[0]

    return np.array([1 / n] * n)


def risk_weighted(covariance_matrix, verbose=0):
    """
    This function computes the weights of the Risk-Weighted strategy.
    :param n*n ndarray covariance_matrix: covariance matrix.
    :param int verbose: Doesn't do anything.
        This argument is passed to provide the same
        signature for all risk-based portfolio strategies.
    :return: vector of weights.
    :rtype: n*1 ndarray
    """

    risks = 1 / np.sqrt(np.diagonal(covariance_matrix))

    return risks / risks.sum()


def maximum_diversification(covariance_matrix, verbose=0):
    """
    This function computes the weights of the Maximum Diversification strategy.
    :param n*n ndarray covariance_matrix: covariance matrix.
    :param int verbose:
        - 0: prints nothing
        - 1: prints balancing dates
        - 2: prints balancing dates and messages of the optimization.
    :return: vector of weights.
    :rtype: n*1 ndarray
    """

    n = covariance_matrix.shape[0]

    def f(x):
        """
        This is the objective function of the Maximum Diversification optimization.
        (cf. page 2 Choueifaty & Coignard)
        :param n*1 ndarray x: vector
        :return: value of the objective function.
        :rtype: float
        Note :
        We use scipy.optimize.minimize to solve the maximization
        problem, thus we have to return the opposite value.
        """

        portfolio_std = (x.T @ covariance_matrix @ x) ** 0.5
        div_ratio = (
            x.T @ np.sqrt(np.diagonal(covariance_matrix * 1e4))
        ) / portfolio_std
        return -div_ratio

    # We now define the constraints on the portfolio weights
    # Defining the inequality constraint -> all weights are positive
    bounds = optimize.Bounds(0.0, 1.0, keep_feasible=True)

    # Defining the equality constraint -> sum of weights = 1
    eq_cons = {
        'type': 'eq',
        'fun': lambda x: np.array([np.sum(x) - 1]),
        'jac': lambda x: np.array([1] * n),
    }

    x0 = np.array([1 / n] * n)
    solution = optimize.minimize(
        f,
        x0,
        method='SLSQP',
        jac=None,
        constraints=eq_cons,
        options={'ftol': 1e-9},
        bounds=bounds
    )

    if verbose == 2:
        print(
            f"Solution message: {solution.message}\nObjective function value: {solution.fun}\n"
        )

    return solution.x


def minimum_variance(covariance_matrix, verbose=0):
    """
    This function computes the weights of the Minimum Variance strategy.
    :param n*n ndarray covariance_matrix: covariance matrix.
    :param int verbose:
        - 0: prints nothing
        - 1: prints balancing dates
        - 2: prints balancing dates and messages of the optimization.
    :return: vector of weights.
    :rtype: n*1 ndarray
    """

    n = covariance_matrix.shape[0]

    def f(x):
        """
        This is the objective function of the Minimum Variance optimization.
        :param n*1 ndarray x: vector
        :return: value of the objective function.
        :rtype: float
        Note:
        Here, the objective function is the standard deviation
        of the portfolio but because the square root is an
        increasing function, we can minimize the variance directly.
        """

        return x.T @ (covariance_matrix * 1e4) @ x

    # We now define the constraints on the portfolio weights
    # Defining the inequality constraint -> all weights are positive
    bounds = optimize.Bounds(0.0, 1.0, keep_feasible=True)

    # Defining the equality constraint -> sum of weights = 1
    eq_cons = {
        'type': 'eq',
        'fun': lambda x: np.array([np.sum(x) - 1]),
        'jac': lambda x: np.array([1] * n),
    }

    x0 = np.array([1 / n] * n)
    solution = optimize.minimize(
        f,
        x0,
        method='SLSQP',
        jac=None,
        constraints=eq_cons,
        options={'ftol': 1e-9},
        bounds=bounds
    )

    if verbose == 2:
        print(
            f"Solution message: {solution.message}\nObjective function value: {solution.fun}\n"
        )

    return solution.x


def equal_risk_contribution(covariance_matrix, verbose=0):
    """
    This function computes the weights of the Equal Risk Contribution strategy.
    :param n*n ndarray covariance_matrix: covariance matrix.
    :param int verbose:
        - 0: prints nothing
        - 1: prints balancing dates
        - 2: prints balancing dates and messages of the optimization.
    :return: vector of weights.
    :rtype: n*1 ndarray
    """

    n = covariance_matrix.shape[0]

    def f(x):
        """
        This is the objective function of the Equal Risk Contribution optimization.
        :param n*1 ndarray x: vector
        :return: value of the objective function.
        :rtype: float
        """

        marginal_risk = (covariance_matrix * 1e4) @ x
        risk_contrib = np.multiply(x, marginal_risk)
        rescaled_variance = 2 * (
            n * np.sum(np.square(risk_contrib)) - (np.sum(risk_contrib)) ** 2
        )
        return rescaled_variance

    # We now define the constraints on the portfolio weights
    # Defining the inequality constraint -> all weights are positive
    bounds = optimize.Bounds(0.0, 1.0, keep_feasible=True)

    # Defining the equality constraint -> sum of weights = 1
    eq_cons = {
        'type': 'eq',
        'fun': lambda x: np.array([np.sum(x) - 1]),
        'jac': lambda x: np.array([1] * n),
    }

    x0 = np.array([1 / n] * n)
    solution = optimize.minimize(
        f,
        x0,
        method='SLSQP',
        jac=None,
        constraints=eq_cons,
        options={'ftol': 1e-9},
        bounds=bounds
    )

    if verbose == 2:
        print(
            f"Solution message: {solution.message}\nObjective function value: {solution.fun}\n"
        )

    return solution.x


PORTFOLIO_FUNCS = {
    'EW': equally_weighted,
    'RW': risk_weighted,
    'MinVar': minimum_variance,
    'MaxDiv': maximum_diversification,
    'ERC': equal_risk_contribution,
}
