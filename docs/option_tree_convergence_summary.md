# Option Tree Convergence Study Summary

- Verified the `ragtop.blackscholes` reference price (~113.34) in the executed notebook.
- Implemented closed-form Black-Scholes pricing helpers and a flexible binomial tree supporting European/American exercise with CRR or JR parameters.
- Demonstrated that the American put exceeds the European put price for representative inputs (6.0864 vs. 5.5635).
- Constructed a convergence dataset for 500-step lattices, plotted log-log errors, and estimated an empirical slope of approximately -0.998, confirming first-order convergence to Black-Scholes.
- Discussed the link between the discrete lattice generator and the continuous Black-Scholes PDE, highlighting \(\mathcal{O}(1/N)\) error decay.
