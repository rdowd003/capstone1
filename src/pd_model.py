# Modeling!
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import matplotlib.pyplot as plt
from utils import XyScaler
from getdata import get_df
from sklearn.metrics import r2_score
import statsmodels.api as sm

#Calculate Mean Square Error
def mse(y, y_hat):
    return np.mean((y  - y_hat)**2)


#Cross validation
#Targets:
# 1. Total Score
# 2. Motor Score
# 3. Change in Score over time


def cross_valid(X, y, base_estimator, n_folds, random_seed=154):
    models = []
    kf = KFold(n_splits=n_folds, shuffle = True, random_state=random_seed)
    test_cv_errors, train_cv_errors = np.empty(n_folds), np.empty(n_folds)
    for idx, (train, test) in enumerate(kf.split(X)):
        # Split into train and test
        X_cv_train, y_cv_train = X[train], y[train]
        X_cv_test, y_cv_test = X[test], y[test]


        # Standardize data.
        standardizer = XyScaler()
        standardizer.fit(X_cv_train, y_cv_train)
        X_cv_train_std, y_cv_train_std = standardizer.transform(X_cv_train, y_cv_train)
        X_cv_test_std, y_cv_test_std = standardizer.transform(X_cv_test, y_cv_test)

        # Fit estimator
        estimator = clone(base_estimator)
        estimator.fit(X_cv_train_std, y_cv_train_std)

        # Measure performance
        y_hat_train = estimator.predict(X_cv_train_std)
        y_hat_test = estimator.predict(X_cv_test_std)

        # Calclate the error metrics
        train_cv_errors[idx] = mse(y_cv_train_std, y_hat_train)
        test_cv_errors[idx] = mse(y_cv_test_std, y_hat_test)

    return train_cv_errors, test_cv_errors


def linear_model(X_train, X_hold, y_train, y_hold):
    ln = LinearRegression()
    linear_cv_errors_train, linear_cv_errors_test = cross_valid(X_train.values, y_train.values, ln, 10)

    linear_mean_cv_errors_train = linear_cv_errors_train.mean(axis=0)
    linear_mean_cv_errors_test = linear_cv_errors_test.mean(axis=0)

    standardizer = XyScaler()
    standardizer.fit(X_train.values, y_train.values)
    X_train_std, y_train_std = standardizer.transform(X_train.values, y_train.values)
    X_hold_std, y_hold_std = standardizer.transform(X_hold.values, y_hold.values)

    final_linear = LinearRegression().fit(X_train_std, y_train_std)
    y_hold_pred_std = final_linear.predict(X_hold_std)
    final_linear_mse = mse(y_hold_std, y_hold_pred_std)
    r2 = r2_score(y_hold_std,y_hold_pred_std)
    print("Linear R2 Score: ",r2)
    print("Final Linear MSE: ",final_linear_mse)
    return (final_linear,y_hold_std,y_hold_pred_std,final_linear_mse)

def lasso_model(X_train, X_hold, y_train, y_hold):
    lasso_alphas = np.logspace(-2,2, num=40)
    lasso_cv_errors_train, lasso_cv_errors_test = train_at_various_alphas(X_train.values, y_train.values, Lasso, lasso_alphas)
    lasso_mean_cv_errors_train = lasso_cv_errors_train.mean(axis=0)
    lasso_mean_cv_errors_test = lasso_cv_errors_test.mean(axis=0)
    lasso_optimal_alpha = get_optimal_alpha(lasso_mean_cv_errors_test)
    plot_mean_CV_error(lasso_mean_cv_errors_train, lasso_mean_cv_errors_test, lasso_alphas, lasso_optimal_alpha,'Optimal Lasso-Alpha Level')
    plt.show()

    standardizer = XyScaler()
    standardizer.fit(X_train.values, y_train.values)
    X_train_std, y_train_std = standardizer.transform(X_train.values, y_train.values)
    X_hold_std, y_hold_std = standardizer.transform(X_hold.values, y_hold.values)
    final_lasso = Lasso(alpha=lasso_optimal_alpha).fit(X_train_std, y_train_std)
    y_hold_pred_std = final_lasso.predict(X_hold_std)
    final_lasso_mse = mse(y_hold_std, y_hold_pred_std)
    r2 = r2_score(y_hold_std,y_hold_pred_std)
    ress = y_hold_std - y_hold_pred_std
    print("Lasso R2 score is: ",r2)
    print("Final Lasso RSS: ",final_lasso_mse)
    print("Optimal Lasso Alpha: ",lasso_optimal_alpha)
    return (final_lasso,y_hold_std,y_hold_pred_std,ress,final_lasso_mse)


def ridge_model(X_train, X_hold, y_train, y_hold):
    ridge_alphas = np.logspace(-2,4, num=40)
    ridge_cv_errors_train, ridge_cv_errors_test = train_at_various_alphas(X_train.values, y_train.values, Ridge, ridge_alphas)
    ridge_mean_cv_errors_train = ridge_cv_errors_train.mean(axis=0)
    ridge_mean_cv_errors_test = ridge_cv_errors_test.mean(axis=0)
    ridge_optimal_alpha = get_optimal_alpha(ridge_mean_cv_errors_test)
    plot_mean_CV_error(ridge_mean_cv_errors_train, ridge_mean_cv_errors_test, ridge_alphas, ridge_optimal_alpha,'Optimal Ridge-Alpha Level')
    plt.show()

    standardizer = XyScaler()
    standardizer.fit(X_train.values, y_train.values)
    X_train_std, y_train_std = standardizer.transform(X_train.values, y_train.values)
    X_hold_std, y_hold_std = standardizer.transform(X_hold.values, y_hold.values)
    final_ridge = Ridge(alpha=ridge_optimal_alpha).fit(X_train_std, y_train_std)
    y_hold_pred_std = final_ridge.predict(X_hold_std)
    final_ridge_mse = mse(y_hold_std, y_hold_pred_std)
    r2 = r2_score(y_hold_std,y_hold_pred_std)
    print("Ridge R2 score is: ",r2)
    print("Final Ridge RSS: ",final_ridge_mse)
    print("Optimal Ridge Alpha: ",ridge_optimal_alpha)
    return (final_ridge,y_hold_std,y_hold_pred_std,final_ridge_mse,X_hold_std)

def get_coefs(model,X):
    df = pd.DataFrame(model.coef_)
    df['coef_names'] = X.columns
    return df

def train_at_various_alphas(X, y, model, alphas, n_folds=10, **kwargs):
    cv_errors_train = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),columns=alphas)
    cv_errors_test = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),columns=alphas)
    for alpha in alphas:
        train_fold_errors, test_fold_errors = cross_valid(X, y, model(alpha=alpha, **kwargs), n_folds=n_folds)
        cv_errors_train.loc[:, alpha] = train_fold_errors
        cv_errors_test.loc[:, alpha] = test_fold_errors
    return cv_errors_train, cv_errors_test


def get_optimal_alpha(mean_cv_errors_test):
    alphas = mean_cv_errors_test.index
    optimal_idx = np.argmin(mean_cv_errors_test.values)
    optimal_alpha = alphas[optimal_idx]
    print(optimal_alpha)
    return optimal_alpha


def plot_mean_CV_error(cv_error_train, cv_error_test, alphas, optimal_alpha,title):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(np.log10(alphas), cv_error_train,label='training set')
    ax.plot(np.log10(alphas), cv_error_test,label='test set')
    ax.axvline(np.log10(optimal_alpha), color='grey')
    ax.set_title(title)
    ax.set_xlabel(r"$\log(\alpha)$")
    ax.legend()
    ax.set_ylabel("MSE")
    plt.show()



df = get_df('parkinsons_data.csv')
#df_est = df.groupby('subject#').head(100)

# Splitting data for test/hold @ ~75 % - hold out subjects 33-42
#______________________________________________________________________________
#Predictor test/train splits (random, by patient, established patient)

#Random model
X = df.drop(columns=['subject#','test_time','motor_UPDRS','total_UPDRS'])
y_total = df['total_UPDRS']
y_motor = df['motor_UPDRS']
X_train, X_hold, y_train_motor, y_hold_motor = train_test_split(X, y_motor, test_size=.1)


#Split, hold off last 11 patients
X_train2 = df[df['subject#']<33].reindex()
X_train2 = X_train2.drop(columns=['subject#','test_time','motor_UPDRS','total_UPDRS'])
X_hold2 = df[df['subject#']>=33].reindex()
X_hold2 = X_hold2.drop(columns=['subject#','test_time','motor_UPDRS','total_UPDRS'])


# Total Score test/train splits
y_train2_total = df[df['subject#']<33]['total_UPDRS']
y_hold2_total = df[df['subject#']>=33]['total_UPDRS']

# Motor Score test/train splits
#y_train_motor = df[df['subject#']<33]['motor_UPDRS']
#y_hold_motor = df[df['subject#']>=33]['motor_UPDRS']


#Established Patient (return) model
a = 0.75 #Percent to take for training
b = 1-a
X_est = df.groupby('subject#').apply(lambda x: x.head(int(len(x) * (a))))
X_train3 = X_est.drop(columns=['subject#','test_time','motor_UPDRS','total_UPDRS'])

X_hold3 = df.groupby('subject#').apply(lambda x: x.tail(int(len(x) * (b))))
X_hold3 = X_hold3.drop(columns=['subject#','test_time','motor_UPDRS','total_UPDRS'])

y_train3_total = df.groupby('subject#')['total_UPDRS'].apply(lambda x: x.head(int(len(x) * (a))))
y_hold3_total = df.groupby('subject#')['total_UPDRS'].apply(lambda x: x.tail(int(len(x) * (b))))



#Running Models
lin_model = linear_model(X_train3,X_hold3,y_train3_total,y_hold3_total)
coef_df_linear = get_coefs(lin_model[0],X_train)
print("Linear coefficients:")
print(coef_df_linear)


las_model = lasso_model(X_train3,X_hold3,y_train3_total,y_hold3_total)
coef_df_lasso = get_coefs(las_model[0],X_train)
print("Lasso coefficients:")
print(coef_df_lasso)

rg_model = ridge_model(X_train3,X_hold3,y_train3_total,y_hold3_total)
coef_df_ridge = get_coefs(rg_model[0],X_train)
print("Ridge coefficients:")
print(coef_df_ridge)


#plot of residuals
plt.scatter(las_model[2],las_model[3])
#plt.xlim([-2,2])
#plt.ylim([-2,2])
plt.xlabel('True Total UPDRS Scores Standardized')
plt.ylabel('Predicted Total UPDRS Scores Standardized')
plt.show()

#Plot coefficients for Lasso & Ridge
X_coef = X_train.columns
ax = plt.subplot(111)
ax.bar(X_coef,coef_df_lasso[0])
ax.bar(X_coef+0.5,coef_df_ridge[0])
ax.set_xlabel('Parameter')
ax.set_ylabel('Standardized Coefficient')
ax.set_title('Standardized coefficients For Lasso & Ridge Regression')
plt.axhline(0, color='blue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()






      #
