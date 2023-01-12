"""
comparecast: a python package for sequentially comparing forecasters
"""

# for convenient imports (`import comparecast as cc; cc.compare_forecasters()`)
from comparecast.cgfs import *
from comparecast.comparecast import *
from comparecast.confint import *
from comparecast.confseq import *
from comparecast.diagnostics import *
from comparecast.eprocess import *
from comparecast.forecasters import *
from comparecast.kernels import *
from comparecast.plotting import *
from comparecast.scoring import *
from comparecast.utils import *

# data submodules (load as, e.g., `cc.data_utils.synthetic.get_data()`)
from comparecast import data_utils
import comparecast.data_utils.synthetic
import comparecast.data_utils.baseball
import comparecast.data_utils.weather
