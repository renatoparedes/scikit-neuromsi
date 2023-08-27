from skneuromsi.utils import storages
from skneuromsi.neural import Cuppini2017
from skneuromsi.mle import AlaisBurr2004
from skneuromsi.cmp import ParameterSweep
from skneuromsi.utils import Bunch
import skneuromsi as skn
import pandas as pd
import numpy as np

from tqdm.notebook import tqdm

import joblib

import xarray as xa

import sys


# sg = storages.DirectoryStorage(10, ".")
# sg[0] = "hola"
# import ipdb; ipdb.set_trace()

# tqdm.pandas()

model = AlaisBurr2004()
res = model.run()

res.to_ndc("zaraza.ndc")

with open("zaraza.ndc", "rb") as fx:
    res = skn.open_ndresult(fx)

# import ipdb; ipdb.set_trace()


res = skn.open_ndrcollection("coso_bz2.ndc", tqdm_cls=None)
# import ipdb; ipdb.set_trace()

# rep = ParameterSweep(
#     model,
#     repeat=1,
#     n_jobs=-1,
#     target="auditory_position",
#     seed=41,
#     storage="memory"
# )

# res = rep.run(noise=True, visual_position=90)

