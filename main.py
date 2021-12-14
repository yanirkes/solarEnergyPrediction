import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from numpy.random import randint
np.random.seed(100465934)
train = pd.read_pickle('trainst1ns16.pkl')
test = pd.read_pickle('testst1ns16.pkl')
