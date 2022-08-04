# %%
import itertools as it
from collections.abc import Iterable
from dataclasses import dataclass

from skneuromsi import paredes2022, alais_burr2004
from skneuromsi import core

import numpy as np

import xarray

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

# %%
@dataclass
class Mock:
    _result: object


# %%
model = paredes2022.Paredes2022()
result = model.run()
stats = Mock(result)

# %%
for n in dir(result.stats):
    if n.startswith("_") or n == "describe":
        continue
    s = result.stats(n)
    print(n, s, type(s))
    print("-" * 30)


# %%
df = result.get_pcoords()
import ipdb

ipdb.set_trace()


a = 1
