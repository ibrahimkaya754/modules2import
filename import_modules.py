# Import Libraries
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.models import Range1d
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure
output_notebook()

TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Dropout, AlphaDropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import BatchNormalization
from livelossplot import PlotLossesKeras
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as k
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, Flatten
# Pandas Libraries
import pandas as pd

# Numpy Libraries
import numpy as np
import random
np.random.seed(8)

# File IO Libraries
import glob
import scipy.io as sio
import pickle

# Plotting Libraries
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.models import Range1d

# Data Preparation Libraries
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
# Scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from tensorflow.keras import backend as K
from keras_contrib.optimizers import Padam, Yogi, ftml
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Lambda
from ipy_table import *
from tensorflow.keras.losses import mean_squared_error

import shutil
current_dir = os.getcwd()+'/'
if 'kerasbackend.py' not in os.listdir('./'):
    print('kerasbackend.py does not exist in the directory, so it will be copied to the working directory')
    shutil.copy('/home/ikaya/anaconda3/envs/py36/lib/python3.6/site-packages/kerasbackend.py',current_dir)
    print('... copied ...')
    from kerasbackend import kerasbackend
    KERASBACKEND = kerasbackend('tensorflow')

else:
    print('kerasbackend.py exist in the directory')
    from kerasbackend import kerasbackend
    KERASBACKEND = kerasbackend('tensorflow')
    
import tensorflow.compat.v1

if KERASBACKEND.KERAS_BACKEND == 'tensorflow':
    # TensorFlow wizardry
    config = tensorflow.compat.v1.ConfigProto() 
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True 
    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 1.0 

