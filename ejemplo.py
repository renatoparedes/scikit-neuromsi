import numpy as np
from pympler import asizeof; import humanize

from skneuromsi.neural import Paredes2022, Cuppini2017
from skneuromsi.mle  import AlaisBurr2004
from skneuromsi.sweep import ParameterSweep

def get_size(obj, sim=1): return humanize.naturalsize(asizeof.asizeof(obj) * sim)

def dictsizes(d):
    for k, v in d.items():
        print(k, f"({type(v)}) ->", get_size(v))


model= Paredes2022()
#res = model.run()


sweep = ParameterSweep(model, target="auditory_position")
reset = sweep.run()
import ipdb; ipdb.set_trace()