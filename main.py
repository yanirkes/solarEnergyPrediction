from timeit import default_timer as timer
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, PredefinedSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
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
validation_indices = np.zeros(y_train.shape[0])
validation_indices[:round(3/4*y_train.shape[0])] = -1
validation_split = PredefinedSplit(validation_indices)

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
# knn (Evri, using dask we were able to run the model in 20-40 % less, check ParallelPostFit)
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
                           , refit=False
                           , error_score='raise'
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
                                        ,'model__estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
                                      }
                                    , cv = validation_split
                                    , refit=False
                                    , scoring='neg_mean_absolute_error'
                                )

start = timer()
gdSearchSvr.fit(x_train,y_train)
train_patched = timer() - start
print(f"time for SVR: {train_patched:.2f} s")
print('SVR best hyperparameters: ', gdSearchSvr.best_params_)

######### Re-fiting the models with the best params ##########
knn = ParallelPostFit(KNeighborsRegressor(n_neighbors=gdSearchKnn.best_params_['model__estimator__n_neighbors']))
knnPipiLine = Pipeline([
    ('standartization',scalar),
    ('model', knn)
    ])
knnPipiLine.fit(x_train,y_train)

DT = ParallelPostFit(DecisionTreeRegressor(min_samples_split = gdSearchDt.best_params_['model__estimator__min_samples_split']
                                           , min_samples_leaf = gdSearchDt.best_params_['model__estimator__min_samples_leaf']
                                           , max_features= gdSearchDt.best_params_['model__estimator__max_features']
                                           , max_depth = gdSearchDt.best_params_['model__estimator__max_depth']))
DtPipiLine = Pipeline([
    ('model', DT)
    ])
DtPipiLine.fit(x_train,y_train)

svr = ParallelPostFit(SVR(kernel = gdSearchSvr.best_params_['model__estimator__kernel']
                          , C = gdSearchSvr.best_params_['model__estimator__C']))
SvrPipiLine = Pipeline([
    ('standartization',scalar),
    ('model', svr)
    ])
SvrPipiLine.fit(x_train,y_train)

######## Models Comparison #####################
print('\nDeafult result')
print('KNN Default MAE: ', round(mean_absolute_error(y_val, knnPipiLineDef.predict(x_val_st)), 4))
print('DT Default MAE: ', round(mean_absolute_error(y_val, dtPipiLineDef.predict(x_val)), 4))
print('SVR Default MAE: ', round(mean_absolute_error(y_val, svrPipiLineDef.predict(x_val_st)), 4))
print('\nHyperparameter optimization modeled result')
print('KNN MAE: ', round(mean_absolute_error(y_val, knnPipiLine.predict(x_val_st)), 4))
print('DT MAE: ', round(mean_absolute_error(y_val, DtPipiLine.predict(x_val)), 4))
print('SVR MAE: ', round(mean_absolute_error(y_val, SvrPipiLine.predict(x_val_st)), 4))

######## DT regressor the best model
x = train.iloc[:, 0:75]
y = train.iloc[:, -1]
final_DT = ParallelPostFit(DecisionTreeRegressor(min_samples_split = gdSearchDt.best_params_['model__estimator__min_samples_split']
                                           , min_samples_leaf = gdSearchDt.best_params_['model__estimator__min_samples_leaf']
                                           , max_features= gdSearchDt.best_params_['model__estimator__max_features']
                                           , max_depth = gdSearchDt.best_params_['model__estimator__max_depth']))
DtFinalPipiLine = Pipeline([
    ('model', final_DT)
    ])
DtFinalPipiLine.fit(x,y)
x_test = test.iloc[:, 0:75]
y_test = test.iloc[:, -1]
print('DT MAE: ', round(mean_absolute_error(y_test, DtFinalPipiLine.predict(x_test)), 4))
print('DT MAPE: ', round(mean_absolute_percentage_error(y_test, DtFinalPipiLine.predict(x_test)), 4))