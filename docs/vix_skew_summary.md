# VIX Skew Summary

- **Valuation context:** Using the 13 October 2025 valuation date, the 0.1806 and 0.2778 year tenors align to roughly 66 and 101 calendar days to expiry (18 December 2025 and 22 January 2026, respectively).
- **Forward levels:** Parity-derived forwards of 20.22 (UXZ5) and 21.05 (UXF6) sit within 4 bps of the corresponding quoted futures, confirming consistent carry assumptions.
- **ATMF volatilities:** UXZ5 carries an 86.0% ATMF volatility versus 77.2% for UXF6, highlighting the near-month’s larger variance premium.
- **Skew behaviour:** UXZ5 displays a sharper downside skew—vols accelerate quickly as log-moneyness turns negative—while UXF6’s skew is flatter through the central deltas and only steepens in the far put wing.
- **Delta structure:** ATM call deltas are tightly clustered near 0.58 for both expiries, but UXZ5’s local deltas collapse faster below the forward, reflecting more aggressive put demand in the near term.

These observations match the notebook’s five-axis visualisations: every expiry plot shows mid-call, mid-put, and Bloomberg vol curves with the same qualitative ordering, reinforcing that the mid-quote inversion delivers a coherent skew read relative to vendor marks.
