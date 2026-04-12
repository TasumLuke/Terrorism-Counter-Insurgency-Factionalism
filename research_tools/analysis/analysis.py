"""
Built this to verify the work I had... I did both computed calculations and non computer calculations

Data loaded from nagaland_mizoram.json.
"""

import json
import math
import sys
import numpy as np
import statsmodels.api as sm
import toolkit as tk


# load everything
with open("nagaland_mizoram.json") as f:
    D = json.load(f)

NAG = D["cases"]["nagaland"]
MIZ = D["cases"]["mizoram"]
W = D["rci_weights"]
NSDP = D["nsdp_2011base"]

alpha, beta, gamma = W["alpha"], W["beta"], W["gamma"]
