from skneuromsi.neural import Cuppini2017
from skneuromsi.cmp import SpatialDisparity
from skneuromsi.utils import Bunch
import pandas as pd
import numpy as np

from tqdm.notebook import tqdm

import joblib

import xarray as xa

import sys

tqdm.pandas()

model = Cuppini2017()

rep = SpatialDisparity(
    model,
    repeat=2,
    n_jobs=-1,
    target="auditory_position",
    seed=41,
    result_storage="disk"
)

res = rep.run(noise=True, visual_position=90)

res[0]

import ipdb; ipdb.set_trace()