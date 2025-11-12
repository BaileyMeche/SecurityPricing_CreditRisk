import json

# Read notebook
with open('notebooks/convertible_bond_ragtop.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix the cell with ragtop import
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        content = ''.join(cell['source'])
        if 'from ragtop.blackscholes import black_scholes' in content and 'Sanity check' in content:
            print(f'Fixing cell at index {i}')
            
            new_code = """import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import ragtop
from ragtop.blackscholes import black_scholes as _ragtop_bs_impl

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

RUN_HEAVY = False      # gate calibration loops and FD grids
SAVE_ARTIFACTS = True
VERBOSE = True

ART_DIR = Path("../artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

# Wrapper to handle ragtop returning either dict or scalar
def _extract_price(result):
    \"\"\"Extract numeric price from ragtop result (handles dict or scalar).\"\"\"
    if isinstance(result, dict):
        for key in ['price', 'value', 'option_price']:
            if key in result:
                return _extract_price(result[key])
        for v in result.values():
            if isinstance(v, (int, float, np.floating)):
                return float(v)
        raise ValueError(f"Cannot extract price from dict: {result}")
    return float(result)

def black_scholes(callput, spot, strike, rate, time, sigma, borrow_cost=0.0):
    \"\"\"Black-Scholes option pricer (wrapped ragtop).\"\"\"
    result = _ragtop_bs_impl(callput, spot, strike, rate, time, sigma, borrow_cost=borrow_cost)
    return _extract_price(result)

# Sanity check: ragtop Black–Scholes signature matches expectations.
_test_px = black_scholes(-1, 625, 612, 0.05, 0.75, 0.62, borrow_cost=0.011)
assert abs(_test_px - 113.34) < 1e-2, f"ragtop black_scholes sanity check failed; got {_test_px}\""""
            
            # Split into lines and add newlines
            cell['source'] = [line + '\n' for line in new_code.split('\n')]
            
            print("Cell fixed!")
            break

# Write back
with open('notebooks/convertible_bond_ragtop.ipynb', 'w', encoding='utf-8') as fw:
    json.dump(nb, fw, indent=1)
    
print("Notebook updated successfully!")
