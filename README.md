# Clay
In this Project we will predict the event of Induced draft fan operation for Coal power plant 
#import libaries

#import libraries
import os
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.dates as mdates 
xformatter = mdates.DateFormatter('%H:%M') # for time axis plots

import sklearn
from scipy.optimize import curve_fit
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.pipeline import Pipeline
from numpy.random import seed
import tensorflow
#from tensorflow import set_random_seed
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)


from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

import warnings
warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings('ignore')
import time
from subprocess import check_output
#import warnings library
import warnings
# ignore all warnings
warnings.filterwarnings('ignore')
# Any results you write to the current directory are saved as output.
df=pd.read_csv('/content/drive/MyDrive/Colab_Notebooks/Project_vibra.csv')
df.head()
