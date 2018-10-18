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
import seaborn as sns

#Calculate Mean Square Error
def mse(y, y_hat):
    return np.mean((y  - y_hat)**2)


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


def linear_model(X_train, X_hold, y_train, y_hold): #you could combine some of this with ridge and lasso making functions for various inner parts:
    ln = LinearRegression()
    linear_cv_errors_train, linear_cv_errors_test = cross_valid(X_train.values, y_train.values, ln, 10)

    linear_mean_cv_errors_train = linear_cv_errors_train.mean(axis=0)
    linear_mean_cv_errors_test = linear_cv_errors_test.mean(axis=0)

    #def standardize(X, y):
    standardizer = XyScaler()
    standardizer.fit(X_train.values, y_train.values)
    X_train_std, y_train_std = standardizer.transform(X_train.values, y_train.values)
    X_hold_std, y_hold_std = standardizer.transform(X_hold.values, y_hold.values)
    #return X_train_std, y_train_std, X_hold_std, y_hold_std

    final_linear = LinearRegression().fit(X_train_std, y_train_std)
    y_hold_pred_std = final_linear.predict(X_hold_std)

    #def calculate_score(y_hold_std, y_hold_pred_std)
    final_linear_mse = mse(y_hold_std, y_hold_pred_std)
    r2 = r2_score(y_hold_std,y_hold_pred_std)
    ress = y_hold_std - y_hold_pred_std
    print("Linear R2 Score: ",r2)
    print("Final Linear MSE: ",final_linear_mse)
    # return y_hold_std,y_hold_pred_std,ress,final_model_mse
    return (final_linear,y_hold_std,y_hold_pred_std,ress,final_linear_mse)

def lasso_model(X_train, X_hold, y_train, y_hold): #could be combined with ridge and lasso through common functions
    lasso_alphas = np.logspace(-2,2, num=40)
    lasso_cv_errors_train, lasso_cv_errors_test = train_at_various_alphas(X_train.values, y_train.values, Lasso, lasso_alphas)
    lasso_mean_cv_errors_train = lasso_cv_errors_train.mean(axis=0)
    lasso_mean_cv_errors_test = lasso_cv_errors_test.mean(axis=0)
    lasso_optimal_alpha = get_optimal_alpha(lasso_mean_cv_errors_test)
    #plot_mean_CV_error(lasso_mean_cv_errors_train, lasso_mean_cv_errors_test, lasso_alphas,
    #lasso_optimal_alpha,'Optimal Lasso-Alpha Level')


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
    #plot_mean_CV_error(ridge_mean_cv_errors_train, ridge_mean_cv_errors_test, ridge_alphas, ridge_optimal_alpha,
    #'Optimal Ridge-Alpha Level')

    standardizer = XyScaler()
    standardizer.fit(X_train.values, y_train.values)
    X_train_std, y_train_std = standardizer.transform(X_train.values, y_train.values)
    X_hold_std, y_hold_std = standardizer.transform(X_hold.values, y_hold.values)
    final_ridge = Ridge(alpha=ridge_optimal_alpha).fit(X_train_std, y_train_std)
    y_hold_pred_std = final_ridge.predict(X_hold_std)
    final_ridge_mse = mse(y_hold_std, y_hold_pred_std)
    r2 = r2_score(y_hold_std,y_hold_pred_std)
    ress = y_hold_std - y_hold_pred_std
    print("Ridge R2 score is: ",r2)
    print("Final Ridge RSS: ",final_ridge_mse)
    print("Optimal Ridge Alpha: ",ridge_optimal_alpha)
    return (final_ridge,y_hold_std,y_hold_pred_std,ress,final_ridge_mse,X_hold_std)

def get_coefs(model,X):
    df = pd.DataFrame(model.coef_)
    df['coef_names'] = X.columns
    return df

def train_at_various_alphas(X, y, model, alphas, n_folds=10, **kwargs): #nice, especially with the kwargs
    cv_errors_train = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),columns=alphas)
    cv_errors_test = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),columns=alphas)
    for alpha in alphas:
        train_fold_errors, test_fold_errors = cross_valid(X, y, model(alpha=alpha, **kwargs), n_folds=n_folds)
        cv_errors_train.loc[:, alpha] = train_fold_errors #I think you can do cv_errors_train[alpha] =  train_fold_errors      instead
        cv_errors_test.loc[:, alpha] = test_fold_errors
    return cv_errors_train, cv_errors_test


def get_optimal_alpha(mean_cv_errors_test): # nice
    alphas = mean_cv_errors_test.index
    optimal_idx = np.argmin(mean_cv_errors_test.values)
    optimal_alpha = alphas[optimal_idx]
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
    plt.tight_layout()
    plt.show()

def save_fig(plot_fxn,name):
    plot_fxn
    plt.savefig(name)


def run_model(model_fxn,X_train,X_hold,y_train,y_hold,model_type,res_filename):
    model = model_fxn(X_train,X_hold,y_train,y_hold)
    coeffs = get_coefs(model[0],X_train)
    return (model,coeffs)
    print(model_type,'Coefficients:')
    print(coeffs)


#Import Data
df = get_df('parkinsons_data.csv')
df['week'] = pd.cut(df['test_time'],bins=27,include_lowest=True)

#Splitting Data Train/Test

#1. Random model
X = df.drop(columns=['subject#','test_time','motor_UPDRS','total_UPDRS','week'])
y_total = df['total_UPDRS']
y_motor = df['motor_UPDRS']
X_train, X_hold, y_train_motor, y_hold_motor = train_test_split(X, y_motor, test_size=.1)


#2. New Patient Model, hold off last 11 patients
X_train2 = df[df['subject#']<33].reindex()
X_train2 = X_train2.drop(columns=['subject#','test_time','motor_UPDRS','total_UPDRS','week'])
X_hold2 = df[df['subject#']>=33].reindex()
X_hold2 = X_hold2.drop(columns=['subject#','test_time','motor_UPDRS','total_UPDRS','week'])

# Total Score test/train splits
y_train2_total = df[df['subject#']<33]['total_UPDRS']
y_hold2_total = df[df['subject#']>=33]['total_UPDRS']

# Motor Score test/train splits
y_train2_motor = df[df['subject#']<33]['motor_UPDRS']
y_hold2_motor = df[df['subject#']>=33]['motor_UPDRS']


#3. Established Patient (return) Model Splits
a = 0.75 #top a% to take for training
b = 1-a  #tail end, b% for test
X_est = df.groupby('subject#').apply(lambda x: x.head(int(len(x) * (a))))  #couldn't you use split_test_train for this?
X_train3 = X_est.drop(columns=['subject#','test_time','motor_UPDRS','total_UPDRS','week'])

X_hold3 = df.groupby('subject#').apply(lambda x: x.tail(int(len(x) * (b))))
X_hold3 = X_hold3.drop(columns=['subject#','test_time','motor_UPDRS','total_UPDRS','week'])

y_train3_total = df.groupby('subject#')['total_UPDRS'].apply(lambda x: x.head(int(len(x) * (a))))
y_hold3_total = df.groupby('subject#')['total_UPDRS'].apply(lambda x: x.tail(int(len(x) * (b))))
y_train3_motor = df.groupby('subject#')['motor_UPDRS'].apply(lambda x: x.head(int(len(x) * (a))))
y_hold3_motor= df.groupby('subject#')['motor_UPDRS'].apply(lambda x: x.tail(int(len(x) * (b))))



linear = run_model(linear_model,X_train3,X_hold3,y_train3_total,y_hold3_total,'Linear Model','lin_res_est2.png')
lasso = run_model(lasso_model,X_train3,X_hold3,y_train3_total,y_hold3_total,'Lasso Model','las_res_est2.png')
ridge = run_model(ridge_model,X_train3,X_hold3,y_train3_total,y_hold3_total,'Ridge Model','rig_res_est2.png')


def res_plot(model):
    fig = plt.figure()
    sns.set_style('whitegrid')
    sns.distplot(model[0][1]-model[0][2],bins=30,hist_kws=dict(edgecolor="k"))
    plt.xlabel('UPDRS Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.show()


def pred_vs_true(model,model_type):
    plt.scatter(model[0][1],model[0][2])
    plt.xlim([-2.5,2.5])
    plt.ylim([-2.5,2.5])
    plt.xlabel('True Standardized Total UPDRS Scores')
    plt.ylabel('Predicted Standardized Total UPDRS Scores')
    plt.title('True vs Predicted Scores: '+ model_type)
    plt.show()


def plot_coefficients(model1,model2,X_train_set,which):
    X_cols = X_train_set.columns
    N=len(X_cols)
    ind = np.arange(N)
    w = 0.5
    model1_coefs = model1[1]
    model2_coefs = model2[1]
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    p1 = ax.bar(ind,model1_coefs[0],w)
    p2 = ax.bar(ind+w,model2_coefs[0],w)
    ax.set_xticks(ind + w / 2)
    ax.set_xticklabels(X_cols)
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Standardized Coefficient')
    ax.set_title('Standardized Lasso and Ridge coefficients: ' + which + ' Patient Model')
    ax.legend((p1[0], p2[0]), ('Lasso', 'Ridge'))
    plt.axhline(0, color='blue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



res_plot(linear)
res_plot(lasso)
res_plot(ridge)

pred_vs_true(linear,'Linear Model')
pred_vs_true(lasso,'Lasso Model')
pred_vs_true(ridge,'Ridge Model')

plot_coefficients(lasso,ridge,X_train2,'New')
plot_coefficients(lasso,ridge,X_train3,'Establish')




#
