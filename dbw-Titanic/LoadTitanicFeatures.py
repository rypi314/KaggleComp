# Databricks notebook source
# MAGIC %sql
# MAGIC 
# MAGIC SELECT * FROM titanic_train limit 20       

# COMMAND ----------

spark.catalog.listTables()

# COMMAND ----------

titanic_train = spark.read.table('titanic_train')
display(titanic_train)

# COMMAND ----------

pddf_titanic_train = titanic_train.toPandas()

# COMMAND ----------

pddf_titanic_train.head()

# COMMAND ----------

# process.py

def loadTitanic(fileString):
    import pandas as pd
    #dfSource = pd.read_csv(fileString)
    dfSource = fileString

    oneHotSex = pd.get_dummies(dfSource.Sex, prefix='Sex')
    oneHotCabin = pd.get_dummies(dfSource.Cabin, prefix='Cabin')
    oneHotEmbarked = pd.get_dummies(dfSource.Embarked, prefix='Embarked')

    dfSource['Age'].fillna(dfSource['Age'].mode()[0], inplace=True)

    dfSource['Fare'].fillna(dfSource['Fare'].mode()[0], inplace=True)

    dfSource['Family'] = dfSource['SibSp'] + dfSource['Parch']

    data = [dfSource]#, dfTitanicTest]
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    for dataset in data:
        # extract titles
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        # replace titles with a more common title or as Rare
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        # convert titles into numbers
        dataset['Title'] = dataset['Title'].map(titles)
        # filling NaN with 0, to get safe
        dataset['Title'] = dataset['Title'].fillna(0)

    import re
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    data = [dfSource]#, test_df]

    for dataset in data:
        dataset['Cabin'] = dataset['Cabin'].fillna("U0")
        dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        dataset['Deck'] = dataset['Deck'].map(deck)
        dataset['Deck'] = dataset['Deck'].fillna(0)
        dataset['Deck'] = dataset['Deck'].astype(int)

    dfFeature = dfSource.join(oneHotSex)
    dfFeature = dfFeature.join(oneHotCabin)
    dfFeature = dfFeature.join(oneHotEmbarked)


    X = dfFeature[['Pclass', 'Age', 'Sex_female', 'Deck', 'Title', 'Fare']]
    X.to_csv('Featured_'+fileString)
    
    try:
        y = dfFeature['Survived']
        y.to_csv('Label_'+fileString)
    except:
        print('Test datset. Missing Label data.')

# COMMAND ----------

loadTitanic(pddf_titanic_train)

# COMMAND ----------

import pandas as pd
#dfSource = pd.read_csv(fileString)
dfSource = pddf_titanic_train

oneHotSex = pd.get_dummies(dfSource.Sex, prefix='Sex')
oneHotCabin = pd.get_dummies(dfSource.Cabin, prefix='Cabin')
oneHotEmbarked = pd.get_dummies(dfSource.Embarked, prefix='Embarked')

dfSource['Age'].fillna(dfSource['Age'].mode()[0], inplace=True)

dfSource['Fare'].fillna(dfSource['Fare'].mode()[0], inplace=True)

dfSource['Family'] = dfSource['SibSp'] + dfSource['Parch']

data = [dfSource]#, dfTitanicTest]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
        # extract titles
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        # replace titles with a more common title or as Rare
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        # convert titles into numbers
        dataset['Title'] = dataset['Title'].map(titles)
        # filling NaN with 0, to get safe
        dataset['Title'] = dataset['Title'].fillna(0)

import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [dfSource]#, test_df]

for dataset in data:
        dataset['Cabin'] = dataset['Cabin'].fillna("U0")
        dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        dataset['Deck'] = dataset['Deck'].map(deck)
        dataset['Deck'] = dataset['Deck'].fillna(0)
        dataset['Deck'] = dataset['Deck'].astype(int)

dfFeature = dfSource.join(oneHotSex)
dfFeature = dfFeature.join(oneHotCabin)
dfFeature = dfFeature.join(oneHotEmbarked)


X = dfFeature[['Pclass', 'Age', 'Sex_female', 'Deck', 'Title', 'Fare']]
X.to_csv('Featured_'+'titantic_train')
    
try:
        y = dfFeature['Survived']
        y.to_csv('Label_'+'fileString')
except:
        print('Test datset. Missing Label data.')

# COMMAND ----------

from collections import Counter
from sklearn.datasets import make_classification

# summarize the dataset
print(X.shape, y.shape)
print(Counter(y))

# COMMAND ----------

print(X)
print(y)

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# COMMAND ----------

# define the multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# COMMAND ----------

# define the model evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# COMMAND ----------

model.fit(X,y)

# COMMAND ----------

model.score(X, y)


# COMMAND ----------

import mlflow
import mlflow.sklearn
import warnings
from urllib.parse import urlparse
import numpy as np
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    with mlflow.start_run():
        
        # define the multinomial logistic regression model
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        
        # define the model evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

        # fit model
        model.fit(X,y)
        
        # find accuracy
        acc = model.score(X, y)
        
        # create confusion matrix
        conf_matx = confusion_matrix(y, model.predict(X))
        true_positive = conf_matx[0][0]
        true_negative = conf_matx[1][1]
        false_positive = conf_matx[0][1]
        false_negative = conf_matx[1][0]

        # model coefficients  
        print('model coefficients:', model.coef_)
        
        # model intercept
        print('model intercept:', model.intercept_)
        
        print('acc:', acc)
        print('confusion matrix', conf_matx)
        print('true_positive', true_positive)
        print('true_negative', true_negative)
        print('false_positive', false_positive)
        print('false_negative', false_negative)
        
        
        mlflow.log_metric("acc", acc)
        mlflow.log_metric("true_positive", true_positive)
        mlflow.log_metric("true_negative", true_negative)
        mlflow.log_metric("false_positive", false_positive)
        mlflow.log_metric("false_negative", false_negative)     
        mlflow.log_metric("model intercept", model.intercept_)        
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name="RegressionTitanticModel")
        else:
            mlflow.sklearn.log_model(model, "model")

