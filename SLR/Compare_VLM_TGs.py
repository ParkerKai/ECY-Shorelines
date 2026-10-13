# -*- coding: utf-8 -*-
"""
Created on 05/01/2026

This script Loads all tide gauge data, removes vertical land motion and reconciles
regional SLR.

@author: kaparker
    WA Dept. of Ecology
    kaip461@ecy.wa.gov
"""

__author__ = "Kai Parker"
__email__ = "kaip461@ecy.wa.gov"

# ===============================================================================
# Import Modules
# ===============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt  # zero-phase, stable
from scipy.stats import t

# ===============================================================================
# Functions
# ===============================================================================

# ===============================================================================
# User Inputs
# ===============================================================================
dir_in_vlm = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\VLM"
dir_out = r"C:\Users\kaip461\OneDrive - Washington State Executive Branch Agencies\Documents\Data\VLM"
os.makedirs(dir_out, exist_ok=True)

# Monthly aggregation method
agg_method = "lowpass"  # Options: 'mean', 'lowpass'

# ===============================================================================
# Tide Gauge Info
# ===============================================================================

# TG = [
#     {"station": '9440910', "name": 'TokePoint',   "vlm_mm_per_yr": 0.035921, "vlm_unc_mm_per_yr": 0.622934},
#     {"station": '9441102', "name": 'Westport',    "vlm_mm_per_yr": 1.632204, "vlm_unc_mm_per_yr": 0.841024},
#     {"station": '9442396', "name": 'LaPush',      "vlm_mm_per_yr": -0.881595, "vlm_unc_mm_per_yr": 0.829389},
#     {"station": '9443090', "name": 'NeahBay',     "vlm_mm_per_yr": 2.80886, "vlm_unc_mm_per_yr": 0.990746},
#     {"station": '9444090', "name": 'PortAngeles', "vlm_mm_per_yr": 1.075373, "vlm_unc_mm_per_yr": 0.53471},
#     {"station": '9444900', "name": 'PortTownsend',"vlm_mm_per_yr": -0.934305, "vlm_unc_mm_per_yr": 0.848552},
#     {"station": '9449880', "name": 'FridayHarbor',"vlm_mm_per_yr": 0.400442, "vlm_unc_mm_per_yr": 0.698111},
#     {"station": '9449424', "name": 'CherryPoint', "vlm_mm_per_yr": 1.254333, "vlm_unc_mm_per_yr": 0.589758},
#     {"station": '9447130', "name": 'Seattle',     "vlm_mm_per_yr": -0.648102, "vlm_unc_mm_per_yr": 0.470093},
#     {"station": '9446484', "name": 'Tacoma',      "vlm_mm_per_yr": -1.278814, "vlm_unc_mm_per_yr": 0.73966},

# ]
# TG_table = pd.DataFrame(TG)V


TG = [
    {"station": "9439099", "name": "Wauna"},
    {"station": "9440569", "name": "Skamokawa"},
    {"station": "9440581", "name": "CapeDisappointment"},
    {"station": "9439040", "name": "Astoria"},
    {"station": "9440910", "name": "TokePoint"},
    {"station": "9441102", "name": "Westport"},
    {"station": "9442396", "name": "LaPush"},
    {"station": "9443090", "name": "NeahBay"},
    {"station": "9444090", "name": "PortAngeles"},
    {"station": "9444900", "name": "PortTownsend"},
    {"station": "9449880", "name": "FridayHarbor"},
    {"station": "9449424", "name": "CherryPoint"},
    {"station": "9447130", "name": "Seattle"},
    {"station": "9446484", "name": "Tacoma"},
]


TG = pd.DataFrame(TG).set_index("station")


# Add VLM
df_vlm = pd.read_csv(
    os.path.join(dir_in_vlm, "vlm_at_TGs_linear.csv"), dtype={"Station": str}
)
df_vlm = df_vlm.set_index("Station")


TG_table = TG.join(df_vlm[["VLM_interp", "VLM_uncertainty"]], how="left")

TG_table = TG_table.rename(
    columns={
        "VLM_interp": "VLM_old",
        "VLM_uncertainty": "vlm_unc_old",
    }
)


path = os.path.join(dir_in_vlm, "2026", "vlm_rates_tidegauges.txt")

df_vlm = pd.read_csv(
    path,
    sep=r"[ \t]+",  # split on runs of spaces or tabs (explicit)
    engine="python",  # more tolerant tokenizer
    encoding="utf-8-sig",  # strips BOM if present at file start
    header=None,
    names=["Station", "Longitude", "Latitude", "VLM_interp", "VLM_uncertainty"],
    comment="#",
    skip_blank_lines=True,
    dtype={
        "Station": "str",  # nullable int (safe if any row is missing)
        "Longitude": "float64",
        "Latitude": "float64",
        "VLM_interp": "float64",
        "VLM_uncertainty": "float64",
    },
)

df_vlm = df_vlm.set_index("Station")


TG_table = TG_table.join(df_vlm[["VLM_interp", "VLM_uncertainty"]], how="left")

TG_table = TG_table.reset_index().rename(columns={"index": "station"})


TG_table = TG_table.rename(
    columns={
        "VLM_interp": "VLM_new",
        "VLM_uncertainty": "vlm_unc_new",
    }
)


TG_table = TG_table[
    ["station", "name", "VLM_old", "VLM_new", "vlm_unc_old", "vlm_unc_new"]
]

TG_table.to_excel(os.path.join(dir_out, "vlm_compare.xlsx"), sheet_name="VLM")
