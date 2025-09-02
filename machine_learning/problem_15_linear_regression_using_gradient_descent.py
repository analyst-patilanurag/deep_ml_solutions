import torch
import numpy as np


def linear_regression_gradient_descent(
    X: np.ndarray, y: np.ndarray, alpha: float, iterations: int
) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (m, n), should include a column of ones
        if an intercept is required.
    y : np.ndarray
        Target vector of shape (m,).
    alpha : float
        Learning rate.
    iterations : int
        Number of gradient descent iterations.

    Returns
    -------
    np.ndarray
        Coefficients (theta) of length n, rounded to 4 decimal places.
    """
    m, n = X.shape
    y = y.reshape(-1, 1)        # ensure column vector (m, 1)
    theta = np.zeros((n, 1))    # initialize parameters (n, 1)

    # Gradient descent loop
    for _ in range(iterations):
        predictions = X @ theta                    # (m, 1)
        errors = predictions - y                   # (m, 1)
        gradient = (X.T @ errors) / m              # (n, 1)
        theta -= alpha * gradient                  # update step

    # Round to 4 decimals
    theta = np.round(theta.flatten(), 4)

    return theta

def linear_regression_gradient_descent(X, y, alpha, iterations) -> torch.Tensor:
    """
    Solve linear regression via gradient descent using PyTorch autograd.
    X: Tensor or convertible of shape (m, n); y: shape (m,) or (m, 1).
    alpha: learning rate; iterations: number of steps.
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    # Ensure tensors and shapes
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    m, n = X_t.shape

    # Parameters with gradients
    theta = torch.zeros((n, 1), dtype=torch.float32, requires_grad=True)

    # Gradient descent loop
    for _ in range(iterations):
        # Forward pass: predictions and MSE loss
        y_pred = X_t @ theta                          # (m, 1)
        loss = ((y_pred - y_t) ** 2).mean()           # Mean Squared Error

        # Backward pass: compute d(loss)/d(theta)
        loss.backward()

        # Parameter update (manual SGD) and grad reset
        with torch.no_grad():
            theta -= alpha * theta.grad
            theta.grad.zero_()

    # Round to 4 decimals and return as 1-D tensor
    with torch.no_grad():
        theta = torch.round(theta * 10000) / 10000

    return theta.detach().squeeze(1)
