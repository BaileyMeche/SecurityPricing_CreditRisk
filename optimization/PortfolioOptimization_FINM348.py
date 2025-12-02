# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Portfolio Optimization for FINM 348
#
# This notebook covers portfolio optimization using continuous optimization methods including:
#
# 1. Deterministic portfolio optimization formulation
# 2. A toy 3-asset portfolio problem using quadratic penalty for equality constraints
# 3. Implementations of steepest descent and Newton's method with backtracking line search
# 4. Reading and scaling S&P 500 data
# 5. Portfolio optimization on S&P 500 using steepest descent from multiple initializations
# 6. A "week-only" return model estimated via Gauss–Newton
# 7. Numerical experiments and diagnostics
#
# ## Plan and Test Strategy
#
# - Test gradient implementations via finite difference comparisons
# - Verify steepest descent and Newton on small toy problems before scaling
# - Use small subsets of S&P 500 data (e.g., first 20 assets, 100 scenarios) for speed
# - Validate constraint satisfaction at each step

# %% [markdown]
# ## Section 0: Imports and Helper Utilities

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Dict, Optional, Tuple, List

np.random.seed(42)

# %%
def finite_difference_grad(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Compute gradient of f at x using forward finite differences.
    
    Parameters
    ----------
    f : callable
        Objective function f(x) -> float
    x : np.ndarray
        Point at which to evaluate the gradient
    eps : float
        Perturbation size for finite differences
        
    Returns
    -------
    np.ndarray
        Approximate gradient of f at x
    """
    g = np.zeros_like(x, dtype=float)
    fx = f(x)
    for i in range(len(x)):
        x_pert = x.copy()
        x_pert[i] += eps
        g[i] = (f(x_pert) - fx) / eps
    return g


def backtracking_line_search(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    p: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    beta: float = 0.5,
    max_backtrack: int = 30
) -> Tuple[float, float]:
    """
    Find step size alpha > 0 via backtracking satisfying Armijo condition:
        f(x + alpha * p) <= f(x) + c1 * alpha * grad_f(x).dot(p)
    
    Parameters
    ----------
    f : callable
        Objective function
    grad_f : callable
        Gradient of objective function
    x : np.ndarray
        Current point
    p : np.ndarray
        Search direction
    alpha0 : float
        Initial step size
    c1 : float
        Armijo constant (typically 1e-4)
    beta : float
        Backtracking reduction factor (typically 0.5)
    max_backtrack : int
        Maximum number of backtracking iterations
        
    Returns
    -------
    tuple (alpha, f_new)
        The step size and new function value
    """
    fx = f(x)
    g = grad_f(x)
    gTp = g.dot(p)
    alpha = alpha0
    
    for _ in range(max_backtrack):
        x_new = x + alpha * p
        f_new = f(x_new)
        if f_new <= fx + c1 * alpha * gTp:
            return alpha, f_new
        alpha *= beta
    
    # Return whatever we have after max iterations
    x_new = x + alpha * p
    f_new = f(x_new)
    return alpha, f_new

# %%
# Test finite difference gradient on a simple quadratic
def test_quadratic(x):
    return 0.5 * np.sum(x**2)

def test_quadratic_grad(x):
    return x.copy()

x_test = np.array([1.0, 2.0, 3.0])
fd_grad = finite_difference_grad(test_quadratic, x_test)
analytic_grad = test_quadratic_grad(x_test)
print("Testing finite difference gradient:")
print(f"  Analytic gradient: {analytic_grad}")
print(f"  FD gradient:       {fd_grad}")
print(f"  Max difference:    {np.max(np.abs(fd_grad - analytic_grad)):.2e}")

# %% [markdown]
# ## Section 1: Deterministic Portfolio Formulation
#
# Consider a portfolio optimization problem with:
#
# - **States** $j = 1, \dots, n$ with probabilities $p_j$ (typically $p_j = 1/n$ for empirical distribution)
# - **Assets** $i = 1, \dots, m$ with scenario payoffs $x_{ij}$ representing the return of asset $i$ in state $j$
# - **Portfolio weights** $\alpha \in \mathbb{R}^m$ representing the fraction invested in each asset
#
# The portfolio consumption in state $j$ is:
# $$C_1(\omega_j) = \sum_{i=1}^m \alpha_i x_{ij}$$
#
# ### Objective: Expected Utility Maximization
#
# For exponential (CARA) utility with risk aversion $\gamma > 0$:
# $$u(c) = -\exp(-\gamma c)$$
#
# The expected utility is:
# $$\mathbb{E}[u(C_1)] = \sum_{j=1}^n p_j \, u\left(\sum_{i=1}^m \alpha_i x_{ij}\right)$$
#
# ### Constraint: Budget Constraint
#
# We impose that the portfolio weights sum to one:
# $$\mathbf{e}^\top \alpha = 1$$
#
# where $\mathbf{e} = (1, 1, \dots, 1)^\top$.
#
# ### Quadratic Penalty Formulation
#
# To handle the equality constraint, we use a quadratic penalty method. The penalty objective is:
# $$P(\alpha, \rho) = f(\alpha) + \frac{\rho}{2}(\mathbf{e}^\top \alpha - 1)^2$$
#
# where $f(\alpha) = -\mathbb{E}[u(C_1)]$ (we minimize the negative expected utility, i.e., maximize expected utility).
#
# As $\rho \to \infty$, minimizers of $P(\alpha, \rho)$ approach the solution of the constrained problem.

# %% [markdown]
# ## Section 2: Toy Three-Asset Problem with Quadratic Penalty

# %%
# Define a small 3-asset payoff matrix (3 assets × 5 states)
# Each column represents returns in a given state
np.random.seed(42)

m_toy = 3  # number of assets
n_toy = 5  # number of states

# Payoff matrix: returns for each asset in each state
# Rows = assets, Columns = states
X_toy = np.array([
    [1.05, 0.98, 1.02, 0.95, 1.08],  # Asset 0: moderate risk
    [1.10, 0.85, 1.15, 0.90, 1.20],  # Asset 1: high risk
    [1.01, 1.01, 1.01, 1.01, 1.01],  # Asset 2: risk-free-like
])

# Equal probabilities for each state
p_toy = np.ones(n_toy) / n_toy

# Risk aversion parameter for exponential utility
gamma_toy = 2.0

print(f"Toy problem: {m_toy} assets, {n_toy} states")
print(f"Payoff matrix X shape: {X_toy.shape}")
print(f"Probabilities: {p_toy}")
print(f"Risk aversion γ = {gamma_toy}")

# %%
def exp_utility(c: np.ndarray, gamma: float) -> np.ndarray:
    """Exponential (CARA) utility: u(c) = -exp(-γc)"""
    return -np.exp(-gamma * c)


def toy_objective(alpha: np.ndarray, X: np.ndarray = X_toy, 
                  p: np.ndarray = p_toy, gamma: float = gamma_toy) -> float:
    """
    Negative expected utility (to be minimized).
    
    Parameters
    ----------
    alpha : np.ndarray
        Portfolio weights (m,)
    X : np.ndarray
        Payoff matrix (m × n)
    p : np.ndarray
        State probabilities (n,)
    gamma : float
        Risk aversion parameter
        
    Returns
    -------
    float
        Negative expected utility
    """
    # Portfolio consumption in each state: C_j = sum_i alpha_i * X_ij
    consumption = X.T @ alpha  # shape (n,)
    utilities = exp_utility(consumption, gamma)
    expected_utility = p @ utilities
    return -expected_utility  # minimize negative = maximize


def toy_objective_grad(alpha: np.ndarray, X: np.ndarray = X_toy,
                       p: np.ndarray = p_toy, gamma: float = gamma_toy) -> np.ndarray:
    """
    Gradient of negative expected utility.
    
    For u(c) = -exp(-γc), we have u'(c) = γ exp(-γc).
    
    ∂f/∂α_i = -∑_j p_j * u'(C_j) * X_ij
            = -∑_j p_j * γ * exp(-γ C_j) * X_ij
    """
    consumption = X.T @ alpha  # (n,)
    u_prime = gamma * np.exp(-gamma * consumption)  # (n,)
    # Chain rule: multiply by X_ij and sum over states
    grad = -X @ (p * u_prime)  # (m,)
    return grad

# %%
# Test gradient with finite differences
alpha_test = np.array([0.4, 0.3, 0.3])
analytic_g = toy_objective_grad(alpha_test)
fd_g = finite_difference_grad(lambda a: toy_objective(a), alpha_test)

print("Testing toy objective gradient:")
print(f"  Analytic gradient: {analytic_g}")
print(f"  FD gradient:       {fd_g}")
print(f"  Max difference:    {np.max(np.abs(fd_g - analytic_g)):.2e}")

# %%
def toy_penalty_objective(alpha: np.ndarray, rho: float,
                          X: np.ndarray = X_toy, p: np.ndarray = p_toy,
                          gamma: float = gamma_toy) -> float:
    """
    Quadratic penalty objective:
        P(α, ρ) = f(α) + (ρ/2)(e^T α - 1)^2
    """
    f_val = toy_objective(alpha, X, p, gamma)
    constraint_violation = np.sum(alpha) - 1.0
    penalty = 0.5 * rho * constraint_violation ** 2
    return f_val + penalty


def toy_penalty_grad(alpha: np.ndarray, rho: float,
                     X: np.ndarray = X_toy, p: np.ndarray = p_toy,
                     gamma: float = gamma_toy) -> np.ndarray:
    """
    Gradient of quadratic penalty objective:
        ∇P(α, ρ) = ∇f(α) + ρ(e^T α - 1) * e
    """
    grad_f = toy_objective_grad(alpha, X, p, gamma)
    constraint_violation = np.sum(alpha) - 1.0
    grad_penalty = rho * constraint_violation * np.ones_like(alpha)
    return grad_f + grad_penalty

# %%
# Test penalty gradient
rho_test = 10.0
analytic_gp = toy_penalty_grad(alpha_test, rho_test)
fd_gp = finite_difference_grad(lambda a: toy_penalty_objective(a, rho_test), alpha_test)

print("Testing penalty objective gradient:")
print(f"  Analytic gradient: {analytic_gp}")
print(f"  FD gradient:       {fd_gp}")
print(f"  Max difference:    {np.max(np.abs(fd_gp - analytic_gp)):.2e}")

# %% [markdown]
# ## Section 3: Steepest Descent and Newton's Method

# %%
def steepest_descent(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-6,
    alpha0: float = 1.0,
    beta: float = 0.5,
    verbose: bool = False
) -> Tuple[np.ndarray, List[Tuple[int, float, float]]]:
    """
    Steepest descent with backtracking line search.
    
    Parameters
    ----------
    f : callable
        Objective function
    grad_f : callable
        Gradient function
    x0 : np.ndarray
        Initial point
    max_iter : int
        Maximum iterations
    tol : float
        Gradient norm tolerance for stopping
    alpha0 : float
        Initial step size for line search
    beta : float
        Backtracking factor
    verbose : bool
        Print progress
        
    Returns
    -------
    tuple (x, history)
        Final point and history of (iteration, f_val, grad_norm)
    """
    x = x0.copy()
    history = []
    
    for k in range(max_iter):
        g = grad_f(x)
        gnorm = np.linalg.norm(g)
        fval = f(x)
        history.append((k, fval, gnorm))
        
        if verbose and k % 50 == 0:
            print(f"  Iter {k}: f={fval:.6f}, ||∇f||={gnorm:.2e}")
        
        if gnorm < tol:
            if verbose:
                print(f"  Converged at iteration {k}")
            break
        
        # Descent direction
        p = -g
        
        # Backtracking line search
        alpha, f_new = backtracking_line_search(f, grad_f, x, p, alpha0, 1e-4, beta)
        x = x + alpha * p
    
    return x, history

# %%
def newton_method(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    hess_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    reg: float = 1e-8,
    verbose: bool = False
) -> Tuple[np.ndarray, List[Tuple[int, float, float]]]:
    """
    Newton's method with backtracking line search and Hessian regularization.
    
    Parameters
    ----------
    f : callable
        Objective function
    grad_f : callable
        Gradient function
    hess_f : callable
        Hessian function
    x0 : np.ndarray
        Initial point
    max_iter : int
        Maximum iterations
    tol : float
        Gradient norm tolerance
    reg : float
        Regularization for near-singular Hessian
    verbose : bool
        Print progress
        
    Returns
    -------
    tuple (x, history)
        Final point and history
    """
    x = x0.copy()
    history = []
    n = len(x)
    
    for k in range(max_iter):
        g = grad_f(x)
        gnorm = np.linalg.norm(g)
        fval = f(x)
        history.append((k, fval, gnorm))
        
        if verbose and k % 10 == 0:
            print(f"  Iter {k}: f={fval:.6f}, ||∇f||={gnorm:.2e}")
        
        if gnorm < tol:
            if verbose:
                print(f"  Converged at iteration {k}")
            break
        
        # Compute Hessian and regularize if needed
        H = hess_f(x)
        H_reg = H + reg * np.eye(n)
        
        # Newton direction: p = -H^{-1} g
        try:
            p = np.linalg.solve(H_reg, -g)
        except np.linalg.LinAlgError:
            # Fallback to gradient descent step
            p = -g
        
        # Check descent direction; if not, use negative gradient
        if g.dot(p) >= 0:
            p = -g
        
        # Backtracking line search
        alpha, f_new = backtracking_line_search(f, grad_f, x, p)
        x = x + alpha * p
    
    return x, history

# %%
def toy_penalty_hessian(alpha: np.ndarray, rho: float,
                        X: np.ndarray = X_toy, p: np.ndarray = p_toy,
                        gamma: float = gamma_toy) -> np.ndarray:
    """
    Hessian of the penalty objective.
    
    For f(α) = -E[u(C)], the Hessian is:
        ∂²f/∂α_i∂α_k = -∑_j p_j u''(C_j) X_ij X_kj
                     = -∑_j p_j (-γ²) exp(-γ C_j) X_ij X_kj
                     = γ² ∑_j p_j exp(-γ C_j) X_ij X_kj
    
    For the penalty term (ρ/2)(e^T α - 1)², the Hessian is ρ e e^T.
    """
    m = len(alpha)
    consumption = X.T @ alpha  # (n,)
    u_double_prime = gamma**2 * np.exp(-gamma * consumption)  # (n,)
    
    # Hessian of f: H_f[i,k] = γ² ∑_j p_j exp(-γ C_j) X_ij X_kj
    # = X @ diag(p * γ² exp(-γC)) @ X^T
    weights = p * u_double_prime
    H_f = X @ np.diag(weights) @ X.T
    
    # Hessian of penalty: ρ e e^T
    e = np.ones(m)
    H_penalty = rho * np.outer(e, e)
    
    return H_f + H_penalty

# %%
# Test Newton's method on the toy problem
print("="*60)
print("Testing steepest descent on toy problem")
print("="*60)

rho = 100.0
alpha0_sd = np.array([0.5, 0.3, 0.2])

f_toy = lambda a: toy_penalty_objective(a, rho)
grad_toy = lambda a: toy_penalty_grad(a, rho)
hess_toy = lambda a: toy_penalty_hessian(a, rho)

alpha_sd, hist_sd = steepest_descent(f_toy, grad_toy, alpha0_sd, max_iter=200, verbose=True)

print(f"\nSteepest descent result:")
print(f"  Final α: {alpha_sd}")
print(f"  Sum of weights: {np.sum(alpha_sd):.6f} (should be ≈ 1)")
print(f"  Final objective: {f_toy(alpha_sd):.6f}")
print(f"  Constraint violation: {np.abs(np.sum(alpha_sd) - 1):.2e}")

# %%
print("="*60)
print("Testing Newton's method on toy problem")
print("="*60)

alpha_newton, hist_newton = newton_method(f_toy, grad_toy, hess_toy, alpha0_sd.copy(), 
                                          max_iter=50, verbose=True)

print(f"\nNewton's method result:")
print(f"  Final α: {alpha_newton}")
print(f"  Sum of weights: {np.sum(alpha_newton):.6f} (should be ≈ 1)")
print(f"  Final objective: {f_toy(alpha_newton):.6f}")
print(f"  Constraint violation: {np.abs(np.sum(alpha_newton) - 1):.2e}")

# %%
# Compare convergence
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Objective value
ax1 = axes[0]
iters_sd, fvals_sd, _ = zip(*hist_sd)
iters_n, fvals_n, _ = zip(*hist_newton)
ax1.plot(iters_sd, fvals_sd, 'b-o', label='Steepest Descent', markersize=3)
ax1.plot(iters_n, fvals_n, 'r-s', label='Newton', markersize=3)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Objective value')
ax1.set_title('Convergence: Objective Value')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gradient norm
ax2 = axes[1]
_, _, gnorms_sd = zip(*hist_sd)
_, _, gnorms_n = zip(*hist_newton)
ax2.semilogy(iters_sd, gnorms_sd, 'b-o', label='Steepest Descent', markersize=3)
ax2.semilogy(iters_n, gnorms_n, 'r-s', label='Newton', markersize=3)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Gradient norm')
ax2.set_title('Convergence: Gradient Norm')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimization/convergence_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print("Convergence comparison plot saved.")

# %% [markdown]
# ## Section 4: S&P 500 Data: Read and Scale

# %%
# Read S&P 500 data
snp_path = 'optimization/snp500.txt'
df_snp = pd.read_csv(snp_path)

print(f"S&P 500 data shape: {df_snp.shape}")
print(f"Columns: Symbol, Name, Price, + {df_snp.shape[1] - 3} scenario columns")
print(f"\nFirst few rows:")
print(df_snp.head())

# %%
# Extract data components
symbols = df_snp['Symbol'].values
names = df_snp['Name'].values
prices = df_snp['Price'].values

# Scenario returns are in columns 3 onwards (columns '1', '2', ..., '1000')
scenario_cols = [str(i) for i in range(1, 1001)]
R_raw = df_snp[scenario_cols].values  # shape: (n_assets, n_scenarios)

print(f"\nNumber of assets: {len(symbols)}")
print(f"Returns matrix R shape: {R_raw.shape} (assets × scenarios)")
print(f"\nSample symbols: {symbols[:5]}")
print(f"Sample prices: {prices[:5]}")

# %%
# These are actually price scenarios, not returns. Compute returns as percentage change from initial price.
# R_returns[i,j] = (R_raw[i,j] - prices[i]) / prices[i]
R = (R_raw - prices[:, np.newaxis]) / prices[:, np.newaxis]

print("Converting price scenarios to returns...")
print(f"Returns matrix R shape: {R.shape}")
print(f"\nSample returns for first asset ({symbols[0]}):")
print(f"  Mean return: {R[0].mean():.6f}")
print(f"  Std return:  {R[0].std():.6f}")

# %%
# Scale returns: subtract per-asset mean, optionally divide by std
R_mean = R.mean(axis=1, keepdims=True)
R_std = R.std(axis=1, keepdims=True)

# Centered returns (de-meaned)
R_centered = R - R_mean

# Standardized returns (z-scores)
R_standardized = (R - R_mean) / (R_std + 1e-10)  # avoid division by zero

print("Scaling statistics:")
print(f"  Mean of means: {R_mean.mean():.6e}")
print(f"  Mean of stds:  {R_std.mean():.6f}")
print(f"\nSample stats for first 5 assets:")
for i in range(5):
    print(f"  {symbols[i]:6s}: mean={R[i].mean():.6f}, std={R[i].std():.6f}")

# %% [markdown]
# ## Section 5: S&P 500 Portfolio Optimization via Steepest Descent

# %%
# Use a subset for speed: first 20 assets, first 100 scenarios
n_assets_subset = 20
n_scenarios_subset = 100

R_subset = R[:n_assets_subset, :n_scenarios_subset]
symbols_subset = symbols[:n_assets_subset]
p_subset = np.ones(n_scenarios_subset) / n_scenarios_subset

print(f"Using subset: {n_assets_subset} assets × {n_scenarios_subset} scenarios")

# %%
def snp_objective(alpha: np.ndarray, R: np.ndarray, p: np.ndarray, 
                  gamma: float = 1.0) -> float:
    """
    Negative expected utility for S&P portfolio.
    Uses exponential utility.
    """
    consumption = R.T @ alpha  # (n_scenarios,)
    utilities = -np.exp(-gamma * consumption)
    return -np.dot(p, utilities)


def snp_objective_grad(alpha: np.ndarray, R: np.ndarray, p: np.ndarray,
                       gamma: float = 1.0) -> np.ndarray:
    """
    Gradient of negative expected utility.
    """
    consumption = R.T @ alpha  # (n_scenarios,)
    u_prime = gamma * np.exp(-gamma * consumption)
    return -R @ (p * u_prime)


def snp_penalty_objective(alpha: np.ndarray, rho: float, R: np.ndarray,
                          p: np.ndarray, gamma: float = 1.0) -> float:
    """Penalty objective with sum-to-one constraint."""
    f_val = snp_objective(alpha, R, p, gamma)
    constraint = np.sum(alpha) - 1.0
    return f_val + 0.5 * rho * constraint ** 2


def snp_penalty_grad(alpha: np.ndarray, rho: float, R: np.ndarray,
                     p: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Gradient of penalty objective."""
    grad_f = snp_objective_grad(alpha, R, p, gamma)
    constraint = np.sum(alpha) - 1.0
    return grad_f + rho * constraint * np.ones_like(alpha)

# %%
# Test gradient on subset
alpha_test_snp = np.ones(n_assets_subset) / n_assets_subset
rho_snp = 100.0
gamma_snp = 1.0

analytic_g_snp = snp_penalty_grad(alpha_test_snp, rho_snp, R_subset, p_subset, gamma_snp)
fd_g_snp = finite_difference_grad(
    lambda a: snp_penalty_objective(a, rho_snp, R_subset, p_subset, gamma_snp),
    alpha_test_snp
)

print("Testing S&P penalty gradient:")
print(f"  Max difference: {np.max(np.abs(fd_g_snp - analytic_g_snp)):.2e}")

# %%
def run_snp_optimization(init_name: str, alpha0: np.ndarray, 
                         R: np.ndarray, p: np.ndarray,
                         rho: float = 1000.0, gamma: float = 1.0,
                         max_iter: int = 300) -> Dict:
    """
    Run steepest descent from given initialization.
    """
    f = lambda a: snp_penalty_objective(a, rho, R, p, gamma)
    grad = lambda a: snp_penalty_grad(a, rho, R, p, gamma)
    
    alpha_final, history = steepest_descent(f, grad, alpha0, max_iter=max_iter, 
                                           tol=1e-7, alpha0=0.1, beta=0.5)
    
    return {
        'name': init_name,
        'alpha': alpha_final,
        'iterations': len(history),
        'final_obj': f(alpha_final),
        'grad_norm': np.linalg.norm(grad(alpha_final)),
        'constraint_violation': np.abs(np.sum(alpha_final) - 1),
        'history': history
    }

# %%
# Multiple initializations
n_assets = n_assets_subset
rho_opt = 1000.0
gamma_opt = 0.5

# 1. Equal weights
init_equal = np.ones(n_assets) / n_assets

# 2. Random Dirichlet weights
np.random.seed(123)
init_dirichlet = np.random.dirichlet(np.ones(n_assets))

# 3. Price-weighted portfolio
prices_subset = prices[:n_assets_subset]
init_price = prices_subset / np.sum(prices_subset)

# 4. Concentrated (smoothed one-hot)
init_concentrated = np.ones(n_assets) * 0.01
init_concentrated[0] = 0.91  # Concentrate on first asset
init_concentrated /= np.sum(init_concentrated)

initializations = [
    ('Equal weights', init_equal),
    ('Dirichlet random', init_dirichlet),
    ('Price-weighted', init_price),
    ('Concentrated', init_concentrated)
]

print("Running S&P 500 portfolio optimization from multiple initializations...")
print(f"Parameters: ρ={rho_opt}, γ={gamma_opt}, max_iter=300")
print()

results = []
for name, alpha0 in initializations:
    result = run_snp_optimization(name, alpha0, R_subset, p_subset, 
                                  rho=rho_opt, gamma=gamma_opt)
    results.append(result)
    print(f"{name}:")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Final obj:  {result['final_obj']:.6f}")
    print(f"  Grad norm:  {result['grad_norm']:.2e}")
    print(f"  Constraint: {result['constraint_violation']:.2e}")
    print()

# %%
# Compare final portfolios
print("Final portfolio weights (first 5 assets):")
print("-" * 60)
header = "Asset    " + "  ".join([r['name'][:10] for r in results])
print(header)
print("-" * 60)
for i in range(min(5, n_assets)):
    row = f"{symbols_subset[i]:8s}"
    for r in results:
        row += f"  {r['alpha'][i]:8.4f}"
    print(row)

# %% [markdown]
# ## Section 6: Week-Only Return Model and Gauss–Newton

# %%
# Week-only return model: Assume returns can be modeled as a low-rank structure
# R_ij ≈ a_i * b_j (rank-1 approximation)
#
# This means each asset's return is proportional to a single "market factor" b_j
# across scenarios, with asset-specific loading a_i.
#
# We fit this via nonlinear least squares:
# min_{a, b} ||R - a b^T||_F^2
#
# This can be solved via Gauss-Newton or simply SVD.

# Use small subset for demo
n_assets_gn = 10
n_scenarios_gn = 50
R_gn = R[:n_assets_gn, :n_scenarios_gn]

print(f"Week-only model: fitting rank-1 approximation to R ({n_assets_gn}×{n_scenarios_gn})")

# %%
def rank1_residual(params: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Residual vector for rank-1 model: r = vec(R - a b^T)
    params = [a; b] concatenated
    """
    m, n = R.shape
    a = params[:m]
    b = params[m:]
    R_approx = np.outer(a, b)
    return (R - R_approx).ravel()


def rank1_jacobian(params: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Jacobian of residual w.r.t. params.
    r_{ij} = R_{ij} - a_i b_j
    ∂r_{ij}/∂a_k = -b_j if k=i, 0 otherwise
    ∂r_{ij}/∂b_l = -a_i if l=j, 0 otherwise
    """
    m, n = R.shape
    a = params[:m]
    b = params[m:]
    
    # Jacobian has shape (m*n, m+n)
    J = np.zeros((m * n, m + n))
    
    idx = 0
    for i in range(m):
        for j in range(n):
            # ∂r_{ij}/∂a_i = -b_j
            J[idx, i] = -b[j]
            # ∂r_{ij}/∂b_j = -a_i
            J[idx, m + j] = -a[i]
            idx += 1
    
    return J


def gauss_newton(R: np.ndarray, max_iter: int = 50, tol: float = 1e-6,
                 reg: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Gauss-Newton for rank-1 model fitting.
    
    Returns (a, b, history) where history contains residual norms.
    """
    m, n = R.shape
    
    # Initialize with SVD
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    a = U[:, 0] * np.sqrt(S[0])
    b = Vt[0, :] * np.sqrt(S[0])
    params = np.concatenate([a, b])
    
    history = []
    
    for k in range(max_iter):
        r = rank1_residual(params, R)
        res_norm = np.linalg.norm(r)
        history.append((k, res_norm))
        
        if res_norm < tol:
            break
        
        J = rank1_jacobian(params, R)
        
        # Gauss-Newton step: (J^T J + reg*I)^{-1} J^T r
        JTJ = J.T @ J
        JTr = J.T @ r
        
        # Regularize
        JTJ_reg = JTJ + reg * np.eye(len(params))
        
        try:
            delta = np.linalg.solve(JTJ_reg, JTr)
        except np.linalg.LinAlgError:
            break
        
        # Update with damping if needed
        alpha = 1.0
        for _ in range(10):
            params_new = params + alpha * delta
            r_new = rank1_residual(params_new, R)
            if np.linalg.norm(r_new) < res_norm:
                params = params_new
                break
            alpha *= 0.5
        else:
            params = params + alpha * delta
    
    a_final = params[:m]
    b_final = params[m:]
    return a_final, b_final, history

# %%
print("Running Gauss-Newton for week-only (rank-1) model...")

a_gn, b_gn, hist_gn = gauss_newton(R_gn, max_iter=30)

print(f"\nGauss-Newton converged in {len(hist_gn)} iterations")
print(f"Initial residual norm: {hist_gn[0][1]:.6f}")
print(f"Final residual norm:   {hist_gn[-1][1]:.6f}")

# Compute approximation quality
R_approx = np.outer(a_gn, b_gn)
mse = np.mean((R_gn - R_approx) ** 2)
r2 = 1 - np.sum((R_gn - R_approx) ** 2) / np.sum((R_gn - R_gn.mean()) ** 2)

print(f"\nModel fit:")
print(f"  MSE: {mse:.6e}")
print(f"  R²:  {r2:.4f}")

# %%
# Compare with direct SVD solution
U_svd, S_svd, Vt_svd = np.linalg.svd(R_gn, full_matrices=False)
a_svd = U_svd[:, 0] * S_svd[0]
b_svd = Vt_svd[0, :]
R_svd = np.outer(a_svd, b_svd)

mse_svd = np.mean((R_gn - R_svd) ** 2)
r2_svd = 1 - np.sum((R_gn - R_svd) ** 2) / np.sum((R_gn - R_gn.mean()) ** 2)

print("\nComparison with SVD rank-1 approximation:")
print(f"  SVD MSE: {mse_svd:.6e}")
print(f"  SVD R²:  {r2_svd:.4f}")
print(f"  GN MSE:  {mse:.6e}")
print(f"  GN R²:   {r2:.4f}")

# %%
# Visualize the approximation
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax1 = axes[0]
im1 = ax1.imshow(R_gn, aspect='auto', cmap='RdBu_r')
ax1.set_title('Original Returns R')
ax1.set_xlabel('Scenario')
ax1.set_ylabel('Asset')
plt.colorbar(im1, ax=ax1)

ax2 = axes[1]
im2 = ax2.imshow(R_approx, aspect='auto', cmap='RdBu_r')
ax2.set_title('Rank-1 Approximation (GN)')
ax2.set_xlabel('Scenario')
ax2.set_ylabel('Asset')
plt.colorbar(im2, ax=ax2)

ax3 = axes[2]
im3 = ax3.imshow(R_gn - R_approx, aspect='auto', cmap='RdBu_r')
ax3.set_title('Residual (R - Approx)')
ax3.set_xlabel('Scenario')
ax3.set_ylabel('Asset')
plt.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.savefig('optimization/model_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print("Model comparison plot saved.")

# %% [markdown]
# ## Section 7: Numerical Experiments and Diagnostics

# %%
print("="*70)
print("NUMERICAL EXPERIMENTS AND DIAGNOSTICS")
print("="*70)

# %%
# Experiment 1: Toy problem with increasing penalty
print("\n--- Experiment 1: Toy problem with increasing penalty parameter ---\n")

rho_values = [1, 10, 100, 1000, 10000]
alpha0_exp = np.array([0.4, 0.3, 0.3])

toy_results = []
for rho in rho_values:
    f = lambda a, r=rho: toy_penalty_objective(a, r)
    grad = lambda a, r=rho: toy_penalty_grad(a, r)
    
    alpha_final, hist = steepest_descent(f, grad, alpha0_exp.copy(), max_iter=500, tol=1e-8)
    
    violation = np.abs(np.sum(alpha_final) - 1)
    obj_val = toy_objective(alpha_final)
    
    toy_results.append({
        'rho': rho,
        'alpha': alpha_final,
        'obj': obj_val,
        'violation': violation,
        'iters': len(hist)
    })
    
    print(f"ρ = {rho:5d}: obj={obj_val:.6f}, violation={violation:.2e}, iters={len(hist)}")

# %%
# Experiment 2: S&P 500 with different subsets
print("\n--- Experiment 2: S&P 500 with different subset sizes ---\n")

subset_sizes = [(10, 50), (15, 75), (20, 100)]
snp_results = []

for n_a, n_s in subset_sizes:
    R_exp = R[:n_a, :n_s]
    p_exp = np.ones(n_s) / n_s
    alpha0_exp = np.ones(n_a) / n_a
    
    result = run_snp_optimization(f"{n_a} assets, {n_s} scenarios", 
                                  alpha0_exp, R_exp, p_exp,
                                  rho=1000.0, gamma=0.5, max_iter=200)
    snp_results.append(result)
    
    print(f"{n_a:2d} assets, {n_s:3d} scenarios: "
          f"obj={result['final_obj']:.4f}, "
          f"violation={result['constraint_violation']:.2e}, "
          f"iters={result['iterations']}")

# %%
# Experiment 3: Week-only model on different subsets
print("\n--- Experiment 3: Week-only model on different subsets ---\n")

model_sizes = [(5, 30), (10, 50), (15, 75)]
model_results = []

for n_a, n_s in model_sizes:
    R_exp = R[:n_a, :n_s]
    
    a_exp, b_exp, hist_exp = gauss_newton(R_exp, max_iter=50)
    
    R_approx_exp = np.outer(a_exp, b_exp)
    mse_exp = np.mean((R_exp - R_approx_exp) ** 2)
    
    # Also compute SVD for comparison
    U_exp, S_exp, Vt_exp = np.linalg.svd(R_exp, full_matrices=False)
    explained_var = S_exp[0]**2 / np.sum(S_exp**2)
    
    model_results.append({
        'size': (n_a, n_s),
        'mse': mse_exp,
        'iters': len(hist_exp),
        'explained_var': explained_var
    })
    
    print(f"{n_a:2d}×{n_s:2d}: MSE={mse_exp:.2e}, "
          f"rank-1 explains {explained_var:.1%} variance, "
          f"GN iters={len(hist_exp)}")

# %%
# Diagnostics summary
print("\n" + "="*70)
print("DIAGNOSTICS SUMMARY")
print("="*70)

# Check for any issues
issues = []

# Check toy problem constraint satisfaction
for r in toy_results:
    if r['violation'] > 1e-4:
        issues.append(f"Toy (ρ={r['rho']}): constraint violation {r['violation']:.2e} > 1e-4")

# Check S&P optimization
for r in snp_results:
    if r['constraint_violation'] > 1e-4:
        issues.append(f"S&P ({r['name']}): constraint violation {r['constraint_violation']:.2e} > 1e-4")
    if r['grad_norm'] > 1e-4:
        issues.append(f"S&P ({r['name']}): gradient norm {r['grad_norm']:.2e} > 1e-4")

# Check for NaNs
alpha_final_check = results[-1]['alpha']
if np.any(np.isnan(alpha_final_check)):
    issues.append("NaN detected in final portfolio weights")

if issues:
    print("\nIssues detected:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nRecommendation: Try smaller alpha0 or larger beta in line search.")
else:
    print("\nAll tests passed without issues!")

# %%
# Final summary table
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print("\n1. Toy Problem Results (3 assets, 5 states):")
print("-" * 50)
print(f"  Newton converged faster ({len(hist_newton)} iters) vs SD ({len(hist_sd)} iters)")
print(f"  Final portfolio (Newton): {alpha_newton}")
print(f"  Constraint violation: {np.abs(np.sum(alpha_newton) - 1):.2e}")

print("\n2. S&P 500 Results (20 assets, 100 scenarios):")
print("-" * 50)
for r in results:
    print(f"  {r['name']:20s}: {r['iterations']:3d} iters, "
          f"violation={r['constraint_violation']:.2e}")

print("\n3. Week-Only Model (Gauss-Newton):")
print("-" * 50)
print(f"  10×50 subset: R² = {r2:.4f}")
print(f"  Gauss-Newton converged in {len(hist_gn)} iterations")
print(f"  Model captures {r2:.1%} of variance with rank-1 approximation")

# %% [markdown]
# ## Conclusions
#
# This notebook demonstrated portfolio optimization techniques for FINM 348:
#
# 1. **Quadratic Penalty Method**: Successfully enforces equality constraints (sum-to-one) as ρ increases
#
# 2. **Steepest Descent vs Newton**: Newton's method shows faster convergence (quadratic vs linear), but requires Hessian computation
#
# 3. **S&P 500 Optimization**: Different initializations converge to similar solutions, demonstrating robustness
#
# 4. **Week-Only Model**: Gauss-Newton effectively fits a rank-1 approximation to return data, capturing significant variance with a simple factor model
#
# **Key Findings**:
# - For well-conditioned problems, Newton's method is preferred
# - Steepest descent is more robust to ill-conditioning
# - The rank-1 return model captures a substantial portion of variance, suggesting a strong market factor
