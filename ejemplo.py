from pympler import asizeof
import humanize

from skneuromsi.neural import Paredes2022, Cuppini2017, Cuppini2014
from skneuromsi.bayesian import Kording2007
from skneuromsi.mle import AlaisBurr2004
from skneuromsi.sweep import ParameterSweep


def get_size(obj, sim=1):
    return humanize.naturalsize(asizeof.asizeof(obj) * sim)


def dictsizes(d):
    for k, v in d.items():
        print(k, f"({type(v)}) ->", get_size(v))


model =  Paredes2022()
# res = model.run()


sweep = ParameterSweep(model, target="auditory_position", n_jobs=15, repeat=2)
result = sweep.run()

import ipdb; ipdb.set_trace()

# import statprof
# statprof.start()
# try:
#     result[0]
# finally:
#     statprof.stop()
#     statprof.display()
