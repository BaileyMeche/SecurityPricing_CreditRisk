"""Black-Scholes pricing utilities mimicking the ragtop API."""
from math import log, sqrt, exp
from scipy.stats import norm


def black_scholes(cp_flag, spot, strike, rate, time, sigma, borrow_cost=0.0):
    """Compute Black-Scholes option price matching the ragtop signature.

    Parameters
    ----------
    cp_flag : int
        +1 for call options, -1 for put options.
    spot : float
        Spot price of the underlying asset.
    strike : float
        Option strike price.
    rate : float
        Risk-free continuously compounded rate.
    time : float
        Time to maturity in years.
    sigma : float
        Volatility of log returns.
    borrow_cost : float, optional
        Continuous dividend yield or borrow cost.
    """
    if time <= 0:
        intrinsic = max(0.0, (spot - strike) if cp_flag == 1 else (strike - spot))
        return intrinsic

    q = borrow_cost
    sqrt_t = sqrt(time)
    d1 = (log(spot / strike) + (rate - q + 0.5 * sigma ** 2) * time) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if cp_flag == 1:
        return exp(-q * time) * spot * norm.cdf(d1) - exp(-rate * time) * strike * norm.cdf(d2)
    if cp_flag == -1:
        return exp(-rate * time) * strike * norm.cdf(-d2) - exp(-q * time) * spot * norm.cdf(-d1)

    raise ValueError("cp_flag must be +1 for calls or -1 for puts")
