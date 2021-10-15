import ipdb
from skneuromsi.alais_burr2004 import AlaisBurr2004, visual_stimulus

model = AlaisBurr2004()

model.run(visual_location=-20, auditory_location=0)
