import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from statsmodels.formula.api import ols
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, RFECV
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import pickle



def ols_test(data):
    outcome = 'weekly_wages'
    predictors = ['age', 'assists_overall', 'penalty_goals', 'penalty_misses', 
               'minutes_played_overall','market_value']
    non_normal = ['age', 'assists_overall', 'penalty_goals', 'penalty_misses', 
               'minutes_played_overall','market_value']
    for feat in non_normal:
        data[feat] = data[feat].map(lambda x: np.log(x + 1))
    pred = '+'.join(predictors)
    formula = outcome + '~' + pred
    model = ols(formula=formula, data=data).fit()
    return model.summary()
    
    
def rank(data,predictors,target):
    linreg = LinearRegression()
    selector = RFE(linreg, n_features_to_select = 6) #for the 6 selected continuous varaiables in the dataset
    selector = selector.fit(predictors, target)
    selector.support_
    selector.ranking_ 
    estimators = selector.estimator_
    
    print("estiamtor coefficients")
    print(estimators.coef_)
    
    
    print("\n")
    
    print("estimator intercepts are")
    print(estimators.intercept_)
          
    print("\n")
    
    print("Are features relevant?")
    
    print("\n")
    
    print(selector.support_)      
         
        
        
def modelling(data, X_train, X_test, y_train, y_test,X_train_cat, X_test_cat):
    ss = StandardScaler()
    X_train_scaled  = ss.fit_transform(X_train)
    
    X_test_scaled = ss.transform(X_test)
    
    linreg = LinearRegression()
    linreg.fit(X_train_scaled, y_train)
    
    print('Baseline model for Continuous variables')
    print('Training r^2:', linreg.score(X_train_scaled, y_train))
    print('Testing r^2:', linreg.score(X_test_scaled, y_test))
    print('Training MSE:', mean_squared_error(y_train, linreg.predict(X_train_scaled)))
    print('Testing MSE:', mean_squared_error(y_test, linreg.predict(X_test_scaled)))
    
    
    print("\n")
        
    #applying log transformations on model to satisfy the assumption of normality
    
    
    X_train_log = np.log(X_train + 1)
    X_test_log = np.log(X_test + 1)
    
    log_X_train_scaled = ss.fit_transform(X_train_log)
    log_X_test_scaled = ss.transform(X_test_log)
    
    
    linreg_2 = LinearRegression()
    linreg_2.fit(log_X_train_scaled, y_train)
    
    print('Baseline model for Continuous variables')
    print('Training r^2:', linreg_2.score(log_X_train_scaled, y_train))
    print('Testing r^2:', linreg_2.score(log_X_test_scaled, y_test))
    print('Training MSE:', mean_squared_error(y_train, linreg_2.predict(log_X_train_scaled)))
    print('Testing MSE:', mean_squared_error(y_test, linreg_2.predict(log_X_test_scaled)))
    
    print("\n")


    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(X_train_cat)
    X_train_ohe = ohe.transform(X_train_cat)
    X_test_ohe = ohe.transform(X_test_cat)
    columns = ohe.get_feature_names(input_features=X_train_cat.columns)
    cat_train_df = pd.DataFrame(X_train_ohe.todense(), columns=columns)
    cat_test_df = pd.DataFrame(X_test_ohe.todense(), columns=columns)
    X_train_all = pd.concat([pd.DataFrame(X_train_scaled), cat_train_df], axis = 1)
    X_test_all = pd.concat([pd.DataFrame(X_test_scaled), cat_test_df], axis = 1)
    linreg_all = LinearRegression()
    linreg_all.fit(X_train_all, y_train)
    
    print("----This is the combination of both continuous and baseline------")
    
    print("\n")
    
    print('Baseline model Continuous and Categorical')
    print('Training r^2:', linreg_all.score(X_train_all, y_train))
    print('Testing r^2:', linreg_all.score(X_test_all, y_test))
    print('Training MSE:', mean_squared_error(y_train, linreg_all.predict(X_train_all)))
    print('Testing MSE:', mean_squared_error(y_test, linreg_all.predict(X_test_all)))
    
    
    poly_features = PolynomialFeatures(2)
  
  # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train_all)

# fit the transformed features to Linear Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
    y_test_predict = poly_model.predict(poly_features.fit_transform(X_test_all))
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predicted))
    r2_train = r2_score(y_train, y_train_predicted)

# evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
    r2_test = r2_score(y_test, y_test_predict)
 

    print("\n")

    print(" Polynomial training set for continuous and categorical")

    print("MSE of training set is {}".format(rmse_train))
    print("R2 score of training set is {}".format(r2_train))

    print("\n")

    print("Polynomial test set for continuous and categorical ")

    print("MSE of test set is {}".format(rmse_test))
    print("R2 score of test set is {}".format(r2_test))

    print("\n")

    lm = LinearRegression()

    scores = cross_val_score(lm, X_train_poly, y_train, cv=10, scoring='r2')
    mse_scores = cross_val_score(lm, X_train_poly, y_train, cv=10, scoring='neg_mean_squared_error')

    print('Cross Validation Mean r2:',np.mean(scores))
    print('Cross Validation Mean MSE:',np.mean(mse_scores))
    print('Cross Validation 10 Fold Score:',scores)
    print ('Cross Validation 10 Fold mean squared error',-(mse_scores) )

    
    
    print("\n")
    
    print("-------Using Extreme gradient boosting to see if we can get better model results----")
          
    print("\n")
    
    print("---Run initial extreme gradient boosting on the algorithm---")
    
    xg_regressor = XGBRegressor(random_state=42)
    xg_regressor.fit(X_train_all, y_train)

    training_preds = xg_regressor.predict(X_train_all)
    test_preds = xg_regressor.predict(X_test_all)

    training_score = r2_score(y_train, training_preds)
    mean_squared_error_train = mean_squared_error (y_train, training_preds)
    test_score = r2_score(y_test, test_preds)
    mean_squared_error_test = mean_squared_error (y_test, test_preds)

    print('Training score: {:.2}'.format(training_score ))
    print('mean_squared_error for train: {:.2}'.format(mean_squared_error_train))
    print('Validation score: {:.2}'.format(test_score ))
    print('mean_squared_error for test: {:.2}'.format(mean_squared_error_test))
    
    
    
    print("\n")
    
    print("----use gridsearch CV to get the best model paramters without model overfitting to the train data---")
    
    xgb1 = XGBRegressor()
    parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [0.01,.03, 0.05, .07], #so called `eta` value
              'max_depth': [2,3,4,5, 6,],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.25],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

    xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 5,
                        n_jobs = -1,
                        verbose=True)
    
    
    
    
    
    xgb_grid.fit(X_train_all, y_train)
    
    
    print(xgb_grid.best_score_)
    print(xgb_grid.best_params_)

    
    
    print("\n")
    
    print("----Running the train and test predictions on the XGB Regressor model----")
    
    xgb_train = xgb_grid.predict(X_train_all)
    xgb_predictions = xgb_grid.predict(X_test_all)


    training_score = r2_score(y_train, xgb_train)
    mean_squared_error_train = mean_squared_error (y_train, xgb_train)

    test_score = r2_score(y_test, xgb_predictions)
    mean_squared_error_test = mean_squared_error (y_test, xgb_predictions)

    print('Training score: {:.2}'.format(training_score ))
    print('mean_squared_error for train: {:.2}'.format(mean_squared_error_train))
    print('test score: {:.2}'.format(test_score ))
    print('mean_squared_error for test: {:.2}'.format(mean_squared_error_test))
    
    
    
    filename = 'finalized_model.sav'
    joblib.dump(xgb_grid, filename)
    
    

