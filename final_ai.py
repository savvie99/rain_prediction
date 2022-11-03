#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:13:21 2022

@author: Savia Laloukiotou
"""

# import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy import stats
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import time
import random
random.seed(30)

#%% READ THE DATASET

df = pd.read_csv('rain.csv')

#%% VIEW THE SHAPE AND HAVE A QUICK LOOK AT THE DATA

print(f'The number of rows are {df.shape[0] } and the number of columns are {df.shape[1]}')
pd.set_option('display.max_columns', None) # to view all the columns
print(df.head()) # to have a quick look at the data

#%%
print(df.info()) # print a summary of the variables in the dataframe

#%%
print(df.describe()) # view a statistics summary of the numeric data

#%%
df.hist(bins=20, figsize=(20,15)) # plot the histograms of numerical data to see the distribution

#%%
df['RainTomorrow'].value_counts().plot(kind='bar') # plot the class variable

#%% print the % of each class
print('\nBalance of positive and negative classes (%):')
print(df['RainTomorrow'].value_counts(normalize=True)*100)

#%% plot the correlation matrix
plt.figure(figsize=(15,15)) # set figure size
ax = sns.heatmap(df.corr(), square=True, annot=True, fmt='.2f') 
ax.set_xticklabels(ax.get_xticklabels(), rotation=90) # set the labels         
plt.show() # show the plot

#%% plot the pairplot for the correlated variables
sns.pairplot( data=df, vars=('MaxTemp','MinTemp','Pressure9am','Pressure3pm', 'Temp9am', 'Temp3pm', 'Evaporation'), hue='RainTomorrow' )
#%% SEPARATE TO NUMERICAL AND CATEGORICAL VARIABLES

# List of categorical variables
categorical = [i for i in df.columns if df[i].dtypes == 'O']
# List of numerical variables
numerical = [i for i in df.columns if i not in categorical]
print('Categorical:\n', categorical, '\n\n', 'Numerical:\n', numerical)

#%%
df_cat= df.select_dtypes(exclude=['float']) # include only object type variables
for col in df_cat.columns:
    print(df_cat[col].unique()) # to print categories name only
    print(df_cat[col].value_counts()) # to print count of every category

#%% VIEW NUMBER AND PERCENTAGE OF MISSING VALUES
print(df.isna().sum())
print(round(100*df.isna().sum()/df.shape[0], 2))

#%% PLOT HEATMAP OF MISSING VALUES
plt.figure(figsize=(12,8))
sns.heatmap(df.isnull())
plt.show()


#%%
# function to replace null numerical values
def replace_numerical(df):
    for col in numerical: # for all numerical columns
        df[col] = df[col].fillna(df[col].median()) # replace missing value with the median
    return df

# function to replace null categorical values
def replace_categorical(df):
    for col in categorical:  # for all categorical columns
        df[col] = df[col].fillna(df[col].mode()[0]) # replace missing value with mode
        
    return df

#%% REPLACE NUMERICAL AND CATEGORICAL VALUES
df = replace_numerical(df)
df = replace_categorical(df)

#%% VIEW PERCENTAGE AND HEATMAP OF MISSING VALUES AGAIN
print(round(100*df.isna().sum()/df.shape[0], 2))
plt.figure(figsize=(12,8))
sns.heatmap(df.isnull(), cbar=False)
plt.show()

#%% REPLACE YES OR NO VALUES WITH 1 AND 0
df = df.replace({'RainTomorrow': {'Yes': 1, 'No': 0}}) # encode raintomorrow
df = df.replace({'RainToday': {'Yes': 1, 'No': 0}}) # encode raintoday

# Check our data frame one more time
print(df.head(5))

#%% DROP THE COLUMN DATE
df.drop('Date', axis=1, inplace = True)

#%% ENCODE THE CATEGORICAL VALUES
le = LabelEncoder() # creating instance of labelencoder
# Assigning numerical values to the columns
df['Location'] = le.fit_transform(df['Location'])
df['WindDir9am'] = le.fit_transform(df['WindDir9am'])
df['WindDir3pm'] = le.fit_transform(df['WindDir3pm'])
df['WindGustDir'] = le.fit_transform(df['WindGustDir'])

# Check our data frame
print(df.head(5))


#%% PLOT OUTLIERS
# function for plotting the outliers with boxplot
def plot_outliers(list):
    sns.set(style="whitegrid")  # set plot style
    plt.figure(figsize=(10, 6)) # set figure size
    sns.boxplot(data=df[list]) # set data to plot
    plt.show() # show the plot
    
#%% plot all the numerical vairables outliers
plot_outliers(['MinTemp','MaxTemp','Temp9am','Temp3pm'])
plot_outliers(['WindGustSpeed','WindSpeed9am','WindSpeed3pm'])
plot_outliers(['Humidity9am','Humidity3pm'])
plot_outliers(['Pressure9am','Pressure3pm'])
plot_outliers(['Cloud9am','Cloud3pm'])
plot_outliers(['Rainfall','Evaporation','Sunshine'])

df_outliers=df[['MinTemp','MaxTemp','Temp9am','Temp3pm','WindGustSpeed',
            'WindSpeed9am','WindSpeed3pm', 'Humidity9am','Humidity3pm',
            'Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Rainfall',
            'Evaporation','Sunshine']]

#%% HANDLE OUTLIERS AND PRINT SHAPE BEFORE AND AFTER

print('Shape of DataFrame Before Removing Outliers', df.shape )
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)] # keep observations with z-score absolute value less than 3
print('Shape of DataFrame After Removing Outliers', df.shape )

#%% PLOT AFTER REMOVING OUTLIERS

plot_outliers(['MinTemp','MaxTemp','Temp9am','Temp3pm'])
plot_outliers(['WindGustSpeed','WindSpeed9am','WindSpeed3pm'])
plot_outliers(['Humidity9am','Humidity3pm'])
plot_outliers(['Pressure9am','Pressure3pm'])
plot_outliers(['Cloud9am','Cloud3pm'])
plot_outliers(['Rainfall','Evaporation','Sunshine'])


#%% split features and target variable
X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']

#%%
# split the original dataframe to train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#%% SCALING THE DATA
# create and fit the scaler on training set
sc=StandardScaler()
X_train=sc.fit_transform(X_train) 
X_test=sc.transform(X_test) # transform the test set

#%%

# Function for fitting and evaluating the models
def fit_evaluate(model, x_train, y_train, x_test, y_test):
    train_start = time.time() # model training starts
    model.fit(x_train, y_train) # the model is fitted on the training set
    train_stop = time.time() # model training stops
    test_start=time.time() # testing starts
    y_pred = model.predict(x_test) # model predicts the labels of the features in the test set
    test_stop=time.time() # testing stops
    print("Training time: ", train_stop - train_start) # print training time
    print("Testing time: ", test_stop - test_start) # print testing time
    print(classification_report(y_test, y_pred)) # print classification report
    
    conf_matrix=confusion_matrix(y_test, y_pred) # calculate confusion matrix
    ax = sns.heatmap(conf_matrix, annot=True, fmt = "g")
    # plot the confusion matrix
    ax.set_title('Confusion Matrix\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');


    ## Display the visualization of the Confusion Matrix.
    plt.show()
    
    
#%% FUNCTION TO CROSS VALIDATE

def cross_validate(model, x_train, y_train):
    scores=cross_val_score(model, x_train, y_train, cv=5) # calculate cross validation scores
    print("Cross Validation Scores:\n")
    print("Mean score: ", scores.mean()) # print the mean
    print("Standard deviation: ", scores.std()) # print standard deviation
    return scores
#%% FUNCTION TO PLOT ROC-AUC CURVE (with help from https://machinelearningmastery.com)
def ROC_AUC(model):
    ns_probs = [0 for _ in range(len(y_test))] # generate a no skill prediction
    lr_probs = model.predict_proba(X_test) # predict probabilities
    lr_probs = lr_probs[:, 1] # keep probabilities for the positive outcome only
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Model: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Model')

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()  
#%% ---- LOGISTIC REGRESSION -------

logreg = LogisticRegression(random_state=42,max_iter=700) # create the classifier
fit_evaluate(logreg, X_train, y_train, X_test, y_test) # fit and evaluate the model
scores_logreg=cross_validate(logreg,X_train, y_train) #  cross validation scores


#%% -------- RANDOM FOREST -----------

rfc=RandomForestClassifier() # create the classifier
fit_evaluate(rfc, X_train, y_train, X_test, y_test) # fit and evaluate the model
scores_rfc=cross_validate(rfc,X_train, y_train) #  cross validation scores

#%% OVERSAMPLE THE MINORITY CLASS AND PRINT THE NEW RATIO
# print number of samples before smote
counter = Counter(y_train)
print('Before SMOTE: ', counter)

# create the SMOTE oversampler
smt = SMOTE()

# fit oversampler on training set
X_smote, y_smote = smt.fit_resample(X_train, y_train)

# print number of samples after smote
counter = Counter(y_smote)
print('After SMOTE: ', counter)

#%%  -------- LOGISTIC REGRESSION ON OVERSAMPLED DATA --------

logreg_res=LogisticRegression(random_state=42,max_iter=700) # create the classifier
fit_evaluate(logreg_res, X_smote, y_smote, X_test, y_test) # fit and evaluate the model
log_res_scores=cross_validate(logreg_res,X_smote, y_smote) # cross validation scores

#%% --------- RANDOM FOREST ON OVERSAMPLED DATA ---------
rfc_res=RandomForestClassifier() # create the classifier
fit_evaluate(rfc_res, X_smote, y_smote, X_test, y_test) # fit and evaluate the model
rfc_res_scores=cross_validate(rfc_res,X_smote, y_smote) # cross validation scores

#%% PCA - FIND OPTIMAL N COMPONENTS
# Create and run a PCA
pca = PCA(n_components=None)
x_scaled = sc.fit_transform(X) 
pca.fit(x_scaled) # fit the pca to scales x values (X is already scaled)

# Get the eigenvalues
print("Eigenvalues:")
print(pca.explained_variance_)
print()

# Get explained variances
print("Variances (Percentage):")
print(pca.explained_variance_ratio_ * 100)
print()

# Make the scree plot
plt.plot(np.cumsum(pca.explained_variance_ratio_ * 100))
plt.xlabel("Number of components (Dimensions)")
plt.ylabel("Explained variance (%)")

#%%
# run pca again with 11 principal components 
x_scaled = sc.fit_transform(X)
pca = PCA(n_components=11)
X_pca = pca.fit_transform(x_scaled)

# Get the transformed dataset
X_pca = pd.DataFrame(X_pca)
print(X_pca.head())
print("\nSize: ")
print(X_pca.shape) # print the size of the dataframe

#%% split PCA dataset to train and test sets
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.20, 
                                                            shuffle=True, random_state=2)
#%% -------- LOGISTIC REGRESSION WITH PRINCIPAL COMPONENTS  -------

logreg_pca = LogisticRegression(random_state=42,max_iter=700) # create the classifier
fit_evaluate(logreg_pca, x_train_pca, y_train_pca, x_test_pca, y_test_pca) # fit and evaluate the model
log_pca=cross_validate(logreg_pca,x_train_pca, y_train_pca) #  cross validation scores

#%% ------- RANDOM FOREST WITH PRINCIPAL COMPONENTS ------
rfc_pca=RandomForestClassifier() # create the classifier
fit_evaluate(rfc_pca, x_train_pca, y_train_pca, x_test_pca, y_test_pca) # fit and evaluate the model
rfc_pca=cross_validate(rfc_pca,x_train_pca, y_train_pca) #  cross validation scores

#%% LOGISTIC REGRESSION GRIDSEARCH CV - 5 fold to speed things up
# load contents from config file that contains the parameters
file="params.config"
contents=open(file).read()
parameters=eval(contents)
# create gridsearch cv instance
grid_search = GridSearchCV(estimator = logreg,  
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)

# fit the gridsearchcv to the training set
grid_search.fit(X_train, y_train)

#%% print the results
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))
# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))

#%% ----- TUNED LOGISTIC REGRESSION ------
log_tuned=LogisticRegression(C=100, max_iter=700, random_state=42)
fit_evaluate(log_tuned, X_train, y_train, X_test, y_test)
scores_log_tuned=cross_validate(log_tuned,X_train, y_train)

#%% RANDOM FOREST WITH RANDOMIZED SEARCH CV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# create RandomizedSearchCV instance
rf_RandomGrid = RandomizedSearchCV(estimator = rfc, param_distributions = param_grid, cv = 10, verbose=2, n_jobs = 4)
# fit to training sets
rf_RandomGrid.fit(X_train, y_train)

#%% print the best parameters

print(rf_RandomGrid.best_params_)

#%% TUNED RANDOM FOREST
rfc_tuned=RandomForestClassifier(n_estimators=72, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=4, bootstrap='True')
fit_evaluate(rfc_tuned, X_train, y_train, X_test, y_test)
scores_rfc_tuned=cross_validate(rfc_tuned,X_train, y_train)

#%% plot the roc-auc for random forest
ROC_AUC(rfc_res)

#%%  plot the roc-auc for logistic regression
ROC_AUC(log_tuned)

#%% plot cross validation scores with boxplots
results=[] # new empty list
results.append(rfc_res_scores) # add random forest scores to list
results.append(scores_log_tuned) # add logistic regression scores to list
fig = plt.figure()
fig.suptitle('Algorithm Comparison on Cross Validation Scores')
plt.boxplot(results)
plt.xticks([1, 2], ['Random Forest', 'Logistic Regression']) # set the labels

#%% print models' cv scores
print("Random forest scores: ", rfc_res_scores)
print("Logistic Regression Scores: ", scores_log_tuned)

























