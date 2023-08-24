from skneuromsi.cmp import storages
from skneuromsi.neural import Cuppini2017
from skneuromsi.cmp import ParameterSweep
from skneuromsi.utils import Bunch
import pandas as pd
import numpy as np

from tqdm.notebook import tqdm

import joblib

import xarray as xa

import sys


# sg = storages.DirectoryStorage(10, ".")
# sg[0] = "hola"
# import ipdb; ipdb.set_trace()

tqdm.pandas()

model = Cuppini2017()
res = model.run()

# rep = ParameterSweep(
#     model,
#     repeat=1,
#     n_jobs=-1,
#     target="auditory_position",
#     seed=41,
#     storage="memory"
# )

# res = rep.run(noise=True, visual_position=90)

import ipdb; ipdb.set_trace()