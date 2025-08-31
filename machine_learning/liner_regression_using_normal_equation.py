import numpy as np
import torch


def linear_regression_normal_equation_numpy(
    X: list[list[float]], y: list[float]
) -> list[float]:
    """
    Solve linear regression using the normal equation with NumPy.

    Parameters
    ----------
    X : list[list[float]]
        Feature matrix of shape (m, n). Should include a column of 1s if
        an intercept term is required.
    y : list[float]
        Target vector of length m.

    Returns
    -------
    list[float]
        Coefficients (theta) of length n, rounded to 4 decimal places.
    """
    # Convert input to NumPy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Normal equation: theta = (X^T X)^(-1) X^T y
    theta = np.linalg.inv(X.T @ X) @ (X.T @ y)

    # Round to 4 decimals
    theta = np.round(theta, 4)

    return theta.tolist()


def linear_regression_normal_equation_torch(
    X: list[list[float]], y: list[float]
) -> list[float]:
    """
    Solve linear regression using the normal equation with PyTorch.

    Parameters
    ----------
    X : list[list[float]]
        Feature matrix of shape (m, n). Should include a column of 1s if
        an intercept term is required.
    y : list[float]
        Target vector of length m.

    Returns
    -------
    list[float]
        Coefficients (theta) of length n, rounded to 4 decimal places.
    """
    # Convert input to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Normal equation: theta = (X^T X)^(-1) X^T y
    theta = torch.linalg.inv(X.T @ X) @ (X.T @ y)

    # Round to 4 decimals (PyTorch doesn’t support decimals directly)
    theta = torch.round(theta * 10000) / 10000

    return theta.tolist()
