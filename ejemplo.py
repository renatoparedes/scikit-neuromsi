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


sweep = ParameterSweep(model, target="auditory_position", n_jobs=-1, repeat=2)
result = sweep.run()



# import statprof
# statprof.start()
# try:
#     for r in result:
#        del r

# finally:
#     statprof.stop()
#     statprof.display()
