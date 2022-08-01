# # Machine learning:

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression , LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, plot_confusion_matrix, confusion_matrix, f1_score
import os
import mlflow
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # to avoid deprecation warnings


dataset=pd.read_csv("clean_dataset2.csv")
dataset.target.replace(["paid_with_delay","paid_in_time"], 1, inplace=True)
dataset.target.replace(["unpaid"], 0, inplace=True)
dataset.rename(columns={"target": "paid"}, inplace=True)
# %%
# Separate target variable Y from features X
target_name = 'paid'

print("Separating labels from features...")
Y = dataset.loc[:,target_name]
X = dataset.loc[:,[c for c in dataset.columns if c!=target_name]] # All columns are kept, except the target
print("...Done.")
print(Y.head())
print()
print(X.head())
print()


# %%
idx = 0
numeric_features = []
numeric_indices = []
categorical_features = []
categorical_indices = []
for i,t in X.dtypes.iteritems():
    if ('float' in str(t)) or ('int' in str(t)) :
        numeric_features.append(i)
        numeric_indices.append(idx)
    else :
        categorical_features.append(i)
        categorical_indices.append(idx)

    idx = idx + 1

print('Found numeric features ', numeric_features,' at positions ', numeric_indices)
print('Found categorical features ', categorical_features,' at positions ', categorical_indices)

# %%
dataset.columns

# %%
# First : always divide dataset into train set & test set !!
print("Dividing into train and test sets...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1, stratify=Y)
# test_size indicates the proportion of rows from X and Y that will go into the test dataset while 
# maintaining the correspondance between the rows from X and Y 

# random_state is an argument that can be found in all functions that have a pseudo-random behaviour
# if random_state is not stated the function will derive a different random result everytime the cell 
# runs, if random_state is given a value the results will be the same everytime the cell runs while
# each different value of radom_state will derive a specific result
print("...Done.")
print()

# %%
# Convert pandas DataFrames to numpy arrays before using scikit-learn
print("Convert pandas DataFrames to numpy arrays...")
X_train = X_train.values
X_test = X_test.values
Y_train = Y_train.tolist()
Y_test = Y_test.tolist()
print("...Done")

print(X_train[0:5,:])
print(X_test[0:2,:])
print()
print(Y_train[0:5])
print(Y_test[0:2])

# %%
# Encoding categorical features and standardizing numerical features
print("Encoding categorical features and standardizing numerical features...")
print()
#print(X_train[0:5,:])

# Normalization
numeric_transformer = StandardScaler()

# OHE / dummyfication
categorical_transformer = OneHotEncoder(drop='first')  #use sparse=False for KNNImputer

featureencoder = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_indices),    
        ('num', numeric_transformer, numeric_indices)
        ]
    )

X_train = featureencoder.fit_transform(X_train)
print("...Done")
#print(X_train[0:5,:])

# Label encoding
print("Encoding labels...")
#print(Y_train[0:5])
encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
print("...Done")
#print(Y_train[0:5])






# %%
dataset.info()

# %%
# Encoding categorical features and standardizing numerical features
print("Encoding categorical features and standardizing numerical features...")
#print(X_test[0:5,:])
X_test = featureencoder.transform(X_test)
print("...Done")
#print(X_test[0:5,:])

# Label encoding
print("Encoding labels...")
#print(Y_test[0:5])
Y_test = encoder.transform(Y_test)
print("...Done")
#print(Y_test[0:5])



# %%
# Set your variables for your environment
EXPERIMENT_NAME="zolo_experiment_v2"

# Instanciate your experiment
client = mlflow.tracking.MlflowClient()
mlflow.set_tracking_uri("https://zolo-app.herokuapp.com")

# Set experiment's info 
mlflow.set_experiment(EXPERIMENT_NAME)

# Get our experiment info
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
#run = client.create_run(experiment.experiment_id) # Creates a new run for a given experiment

# Call mlflow autolog
mlflow.sklearn.autolog(log_post_training_metrics= True)

with mlflow.start_run():
    # Specified Parameters 
    params = {'C': [0.05, 0.1, 0.2, 0.3, 0.5, 1, 2], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']}


    # Instanciate and fit the model 
    classifier = LogisticRegression()


    gridsearch = GridSearchCV(classifier, params, scoring="f1", refit = True, n_jobs=-1, cv=10)# cv : the number of folds to be used for CV
    gridsearch.fit(X_train, Y_train)

    #confusion matrix
    confusion_matrix_train = confusion_matrix(Y_train, (gridsearch.predict(X_train)))
    confusion_matrix_test = confusion_matrix(Y_test, (gridsearch.predict(X_test)))

    test_true_negative = confusion_matrix_test[0][0]
    test_true_positive = confusion_matrix_test[1][1]
    test_false_positive = confusion_matrix_test[0][1]
    test_false_negative = confusion_matrix_test[1][0]

    train_true_negative = confusion_matrix_train[0][0]
    train_true_positive = confusion_matrix_train[1][1]
    train_false_positive = confusion_matrix_train[0][1]
    train_false_negative = confusion_matrix_train[1][0]

    #score to predict 
    train_score_roc_auc=roc_auc_score(Y_train, gridsearch.predict_proba(X_train)[:,1])
    test_score_roc_auc=roc_auc_score(Y_test, gridsearch.predict_proba(X_test)[:,1])
    train_f1_score=f1_score(Y_train, (gridsearch.predict(X_train)), average='binary')
    test_f1_score=f1_score(Y_test, (gridsearch.predict(X_test)), average='binary')

    #print score
    print("train roc_auc score=:",train_score_roc_auc)
    print("test roc_auc score=:",test_score_roc_auc)
    print("train f1 score=:",train_f1_score)
    print("test f1score=:",test_f1_score)


    # Log Metric 
    mlflow.log_metric("train roc_auc score", train_score_roc_auc)
    mlflow.log_metric("test roc_auc score", test_score_roc_auc)
    mlflow.log_metric("train f1 score", train_f1_score)
    mlflow.log_metric("test f1 score", test_f1_score)


    mlflow.log_metric("train_true_positive", train_true_positive)
    mlflow.log_metric("train_true_negative", train_true_negative)
    mlflow.log_metric("train_false_positive", train_false_positive)
    mlflow.log_metric("train_false_negative", train_false_negative)

    mlflow.log_metric("test_true_positive", test_true_positive)
    mlflow.log_metric("test_true_negative", test_true_negative)
    mlflow.log_metric("test_false_positive", test_false_positive)
    mlflow.log_metric("test_false_negative", test_false_negative)

    # Log model 
    mlflow.sklearn.log_model(gridsearch, "zolo-models_v2")

