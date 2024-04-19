import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import joblib
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from joblib import load
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

x_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')

x_test = pd.read_csv('x_test.csv')
y_test = pd.read_csv('y_test.csv')



































