# MLML-occupancy_prediction
All folders containing different methods implemented for the project.
for running the script following required libraries has to be installed and import in python.
python version 3.9

Using following imports:

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.layers import Dropout, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
