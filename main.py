from timeit import default_timer as timer
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, PredefinedSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from dask_ml.wrappers import ParallelPostFit


np.random.seed(100465934)
train = pd.read_pickle('trainst1ns16.pkl')
test = pd.read_pickle('testst1ns16.pkl')
train_close, test_close = train.iloc[:,0:75], test.iloc[:,0:75]

##### split 10 years train and 2 year val ########
x_train, x_val, y_train, y_val = train_test_split(train_close.values,pd.Series(train['energy']),train_size=365*10)

##### train models ############
scalar = StandardScaler().fit(x_val)
x_val_st = scalar.transform(x_val)

# Predefined Val Split
validation_idx = np.repeat(-1, y_train.shape)
validation_idx[np.random.choice(validation_idx.shape[0],
       int(round(.2*validation_idx.shape[0])), replace = False)] = 0

# Now, create a list which contains a single tuple of two elements,
# which are arrays containing the indices for the development and
# validation sets, respectively.
validation_split = list(PredefinedSplit(validation_idx).split())

# sanity check
print(len(validation_split[0][0]))
print(len(validation_split[0][0])/float(validation_idx.shape[0]))
print(validation_idx.shape[0] == y_train.shape[0])
print(set(validation_split[0][0]).intersection(set(validation_split[0][1])))

#### Defaulted Models
# knn
scalar = StandardScaler()
knn = KNeighborsRegressor()
knnPipiLineDef = Pipeline([
    ('standartization',scalar),
    ('model', knn)
    ])
knnPipiLineDef.fit(x_train,y_train)


# Deision tree
scalar = StandardScaler()
dt = DecisionTreeRegressor()
dtPipiLineDef = Pipeline([
    ('standartization',scalar),
    ('model', dt)
    ])
dtPipiLineDef.fit(x_train,y_train)

# svm
scalar = StandardScaler()
svr = SVR()
svrPipiLineDef = Pipeline([
    ('standartization',scalar),
    ('model', svr)
    ])
svrPipiLineDef.fit(x_train,y_train)

########################################## DASK ######################################
#### Non Defaulted Defaulted Models
# knn (Evri using dask we were able to run the model in 20-40 % less, check ParallelPostFit)
# knn
print('\nKNN\n')
scalar = StandardScaler()
knn = ParallelPostFit(KNeighborsRegressor())
knnPipiLine = Pipeline([
    ('standartization',scalar),
    ('model', knn)
    ])
gdSearchKnn = GridSearchCV(knnPipiLine
                           , {'model__estimator__n_neighbors':[1,2,3,4,5,6,7,8]}
                           , cv = validation_split
                           ,refit=False
                           , scoring='neg_mean_absolute_error')

start = timer()
gdSearchKnn.fit(x_train,y_train)
train_patched = timer() - start
print(f"time for KNN: {train_patched:.2f} s")
print('KNN best hyperparameters: ', gdSearchKnn.best_params_)

print('\nDT\n')
# Deision tree
dt = ParallelPostFit(DecisionTreeRegressor())
dtPipiLine = Pipeline([
    ('model', dt)
    ])
gdSearchDt = RandomizedSearchCV(dtPipiLine
                                    , {'model__estimator__max_depth':[2, 5, 10, 30, 75, 100]
                                        , 'model__estimator__min_samples_split': [2, 5, 10, 20, 50]
                                        , 'model__estimator__max_features': ['auto','sqrt']
                                        , 'model__estimator__min_samples_leaf': [2,4,10]
                                        , 'model__estimator__criterion': ['MAE']
                                      }
                                    , cv = validation_split
                                    , refit=False
                                    , scoring='neg_mean_absolute_error'
                                )

start = timer()
gdSearchDt.fit(x_train,y_train)
train_patched = timer() - start
print(f"time for DT: {train_patched:.2f} s")
print('DT best hyperparameters: ', gdSearchDt.best_params_)

print('\nSVM\n')
# svm
scalar = StandardScaler()
svr = ParallelPostFit(SVR())
svrPipiLine = Pipeline([
    ('standartization',scalar),
    ('model', svr)
    ])
gdSearchSvr = RandomizedSearchCV(svrPipiLine
                                    , {'model__estimator__C':[1,2,3,4]
                                        ,'model__estimator__kernel': ['linear', 'rbf']
                                      }
                                    , cv = validation_split,refit=False
                                    , scoring='neg_mean_absolute_error'
                                )

start = timer()
gdSearchSvr.fit(x_train,y_train)
train_patched = timer() - start
print(f"time for SVR: {train_patched:.2f} s")
print('SVR best hyperparameters: ', gdSearchSvr.best_params_)

######### Re-fiting the models with the best params ##########
knn = ParallelPostFit(KNeighborsRegressor(n_neighbors=8))
knnPipiLine = Pipeline([
    ('standartization',scalar),
    ('model', knn)
    ])
knnPipiLine.fit(x_train,y_train)

DT = ParallelPostFit(DecisionTreeRegressor(min_samples_split=10, min_samples_leaf = 10, max_features= 'auto', max_depth = 10))
DtPipiLine = Pipeline([
    ('model', DT)
    ])
DtPipiLine.fit(x_train,y_train)

svr = ParallelPostFit(SVR(kernel = 'linear', C = 3))
SvrPipiLine = Pipeline([
    ('standartization',scalar),
    ('model', svr)
    ])
SvrPipiLine.fit(x_train,y_train)

######## Models Comparison #####################
print('\nDeafult result')
print('KNN Default MSE: ', round(mean_absolute_error(y_val, knnPipiLineDef.predict(x_val_st)), 4))
print('DT Default MSE: ', round(mean_absolute_error(y_val, dtPipiLineDef.predict(x_val)), 4))
print('SVR Default MSE: ', round(mean_absolute_error(y_val, svrPipiLineDef.predict(x_val_st)), 4))
print('\nHyperparameter optimization modeled result')
print('KNN MSE: ', round(mean_absolute_error(y_val, knnPipiLine.predict(x_val_st)), 4))
print('DT MSE: ', round(mean_absolute_error(y_val, DtPipiLine.predict(x_val)), 4))
print('SVR MSE: ', round(mean_absolute_error(y_val, SvrPipiLine.predict(x_val_st)), 4))

######## DT regressor the best model
x = train.iloc[:, 0:75]
y = train.iloc[:, -1]
final_DT = ParallelPostFit(DecisionTreeRegressor(min_samples_split=10, max_features= 'sqrt', max_depth = 5))
DtFinalPipiLine = Pipeline([
    ('model', final_DT)
    ])
DtFinalPipiLine.fit(x,y)
x_test = test.iloc[:, 0:75]
y_test = test.iloc[:, -1]
print('DT MSE: ', round(mean_absolute_error(y_test, DtFinalPipiLine.predict(x_test)), 4))