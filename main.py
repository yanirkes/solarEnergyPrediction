import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from numpy.random import randint
import matplotlib.pyplot as plt

np.random.seed(100465934)
train = pd.read_pickle('trainst1ns16.pkl')
test = pd.read_pickle('testst1ns16.pkl')

######### visualization #########
train_close, test_close = train.iloc[:,0:75], test.iloc[:,0:75]
train_close.hist()

######### visualization #########
scalar = StandardScaler().fit(train_close)
train_close_st = scalar.transform(train_close)
pca = PCA(n_components=6)
pca.fit(train_close_st)


plt.plot(range(0,len(pca.explained_variance_)), pca.explained_variance_)
plt.ylabel('Explained Variance')
plt.xlabel('Principal Components')
plt.xticks(range(0,len(pca.explained_variance_)),
           ["1st comp", "2nd comp", "3rd comp", "4rd comp", "5rd comp", "6rd comp" ], rotation=60)
plt.title('Explained Variance Ratio')
plt.show()

print("1 to the 6th components explained_variance_ratio_: ", sum(pca.explained_variance_ratio_[0:6]))

######## plot most important columns by components #########

fig, axs = plt.subplots(6,6)
axs[1, 0].scatter(pca.components_[:, 0],pca.components_[:, 1])
axs[1, 0].set_title('Axis [1,0]')
axs[2, 0].scatter(pca.components_[:, 0],pca.components_[:, 2])
axs[2, 0].set_title('Axis [2,0]')
axs[3, 0].scatter(pca.components_[:, 0],pca.components_[:, 3])
axs[3, 0].set_title('Axis [3,0]')
axs[4, 0].scatter(pca.components_[:, 0],pca.components_[:, 4])
axs[4, 0].set_title('Axis [4,0]')
axs[5, 0].scatter(pca.components_[:, 0],pca.components_[:, 5])
axs[5, 0].set_title('Axis [4,0]')

axs[2, 1].scatter(pca.components_[:, 1],pca.components_[:, 2])
axs[2, 1].set_title('Axis [2,1]')
axs[3, 1].scatter(pca.components_[:, 1],pca.components_[:, 3])
axs[3, 1].set_title('Axis [3,1]')
axs[4, 1].scatter(pca.components_[:, 1],pca.components_[:, 4])
axs[4, 1].set_title('Axis [4,1]')
axs[5, 1].scatter(pca.components_[:, 1],pca.components_[:, 5])
axs[5, 1].set_title('Axis [5,1]')

axs[3,2].scatter(pca.components_[:, 2],pca.components_[:, 3])
axs[3,2].set_title('Axis [3,2]')
axs[4,2].scatter(pca.components_[:, 2],pca.components_[:, 4])
axs[4,2].set_title('Axis [4,2]')
axs[5,2].scatter(pca.components_[:, 2],pca.components_[:, 5])
axs[5,2].set_title('Axis [5,2]')

axs[4,3].scatter(pca.components_[:, 3],pca.components_[:, 4])
axs[4,3].set_title('Axis [4,3]')
axs[5,3].scatter(pca.components_[:, 3],pca.components_[:, 5])
axs[5,3].set_title('Axis [5,3]')

axs[5, 4].scatter(pca.components_[:, 4],pca.components_[:, 5])
axs[5, 4].set_title('Axis [5,4]')

##### split train and val ########
train_after_pca = pd.DataFrame(columns=train.columns[0:6],data= pca.fit_transform(train_close_st))


x_train, x_val, y_train, y_val = train_test_split(train_after_pca.values,pd.Series(train['energy']))


##### train models ############
knn = KNeighborsRegressor(5).fit(x_train, y_train)
dt  = DecisionTreeRegressor().fit(x_train,y_train)
svm = SVR().fit(x_train,y_train)

# intial score
knn.score(x_val, y_val)
# 0.7857

dt.score(x_val, y_val)
# 0.6579

svm.score(x_val, y_val)
# 0.0009