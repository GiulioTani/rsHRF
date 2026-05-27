import numpy as np
import sys

default_parameters = {}
default_parameters["estimation"] = "canon2dd"  # why?
default_parameters["passband"] = [0.01, 0.08]  # why?
default_parameters["passband_deconvolve"] = [0.0, sys.float_info.max]  # why?
default_parameters["TR"] = -1
default_parameters["localK"] = 1  # why?
default_parameters["T"] = 3  # why?
default_parameters["T0"] = 1  # why?
default_parameters["TD_DD"] = 2  # why?
default_parameters["AR_lag"] = 1  # why?
default_parameters["thr"] = 1  # why?
default_parameters["order"] = 3  # why?
default_parameters["volterra"] = 0  # why?
default_parameters["len"] = 24  # why?
default_parameters["temporal_mask"] = []  # why?
default_parameters["min_onset_search"] = 4  # why?
default_parameters["max_onset_search"] = 8  # why?
default_parameters["wiener"] = False
default_parameters["dt"] = -1
default_parameters["lag"] = np.arange(
    0,
    0,
    dtype="int",
)

available_estimations = ["canon2dd", "sFIR", "FIR", "fourier", "hanning", "gamma"]
