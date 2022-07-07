# Databricks notebook source
# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    #alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    #l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    alpha = .8
    l1_ratio = .1
    
    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")

# COMMAND ----------

test_x.to_csv('/dbfs/FileStore/Sample.csv')

# COMMAND ----------

train_x.head()

# COMMAND ----------

train_x.describe()

# COMMAND ----------

# load input data table as a Spark DataFrame
input_data = spark.table(df)
model_udf = mlflow.pyfunc.spark_udf(model_path)
df = input_data.withColumn("prediction", model_udf())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### JSON Example
# MAGIC 
# MAGIC [
# MAGIC {
# MAGIC "fixed acidity": 5.5,
# MAGIC "volatile acidity": 0.78,
# MAGIC "citric acid": 0.01,
# MAGIC "residual sugar": 2.2,
# MAGIC "chlorides": 0.045,
# MAGIC "free sulfur dioxide": 45.0,
# MAGIC "total sulfur dioxide": 32.0,
# MAGIC "density": 0.99257,
# MAGIC "pH": 0.54,
# MAGIC "sulphates": 0.58,
# MAGIC "alcohol": 12.6
# MAGIC }
# MAGIC ]

# COMMAND ----------

import numpy as np
import pandas as pd
import requests

MODEL_VERSION_URI ='https://adb-8582279356896427.7.azuredatabricks.net/model/ElasticnetWineModel/Production/invocations'
DATABRICKS_API_TOKEN = 'dapidcc67a51fa75f150663f69f47cf7c8ce-2'

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(model_uri, databricks_token, data):
  headers = {
    "Authorization": f"Bearer {databricks_token}",
    "Content-Type": "application/json",
  }
  data_json = data.to_dict(orient='records') if isinstance(data, pd.DataFrame) else create_tf_serving_json(data)
  response = requests.request(method='POST', headers=headers, url=model_uri, json=data_json)
  if response.status_code != 200:
      raise Exception(f"Request failed with status {response.status_code}, {response.text}")
  return response.json()

# Scoring a model that accepts pandas DataFrames
data =  pd.DataFrame([{
"fixed acidity": 5.5,
"volatile acidity": 0.78,
"citric acid": 0.01,
"residual sugar": 2.2,
"chlorides": 0.045,
"free sulfur dioxide": 45.0,
"total sulfur dioxide": 32.0,
"density": 0.99257,
"pH": 0.54,
"sulphates": 0.58,
"alcohol": 12.6
}])

# Score the data
score_model(MODEL_VERSION_URI, DATABRICKS_API_TOKEN, data)

# COMMAND ----------

import numpy as np
import pandas as pd
import requests

MODEL_VERSION_URI ='https://adb-8582279356896427.7.azuredatabricks.net/model/ElasticnetWineModel/Production/invocations'
DATABRICKS_API_TOKEN = 'dapidcc67a51fa75f150663f69f47cf7c8ce-2'

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(model_uri, databricks_token, data):
  headers = {
    "Authorization": f"Bearer {databricks_token}",
    "Content-Type": "application/json",
  }
  data_json = data.to_dict(orient='records') if isinstance(data, pd.DataFrame) else create_tf_serving_json(data)
  response = requests.request(method='POST', headers=headers, url=model_uri, json=data_json)
  if response.status_code != 200:
      raise Exception(f"Request failed with status {response.status_code}, {response.text}")
  return response.json()

# Scoring a model that accepts pandas DataFrames
data =  df.to_json()

# Score the data
score_model(MODEL_VERSION_URI, DATABRICKS_API_TOKEN, data)

# COMMAND ----------

from mlflow import spark
modelSpark = mlflow.spark.load_model("https://adb-8582279356896427.7.azuredatabricks.net/model/ElasticnetWineModel/Production/invocations")
# Prepare test documents, which are unlabeled (id, text) tuples.
#test = spark.createDataFrame([
#    (4, "spark i j k"),
#    (5, "l m n"),
#    (6, "spark hadoop spark"),
#    (7, "apache hadoop")], ["id", "text"])
# Make predictions on test documents
prediction = model.transform(test)

# COMMAND ----------

import mlflow

# COMMAND ----------

model_path = 'models:/ElasticnetWineModel/3'

# COMMAND ----------

display(df)

# COMMAND ----------

df.show()

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_path)
model.predict(df)

# COMMAND ----------

model

# COMMAND ----------

from mlflow import spark

with mlflow.start_run() as active_run:
  model.predict(df)


# COMMAND ----------

#from mlflow import spark
#model = mlflow.spark.load_model("spark-model")
# Prepare test documents, which are unlabeled (id, text) tuples.
test = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")], ["id", "text"])
# Make predictions on test documents
prediction = model.transform(test)

# COMMAND ----------

# load input data table as a Spark DataFrame
input_data = spark.table(df)
model_udf = mlflow.pyfunc.spark_udf(model_path)
df1 = input_data.withColumn("prediction", model_udf())

# COMMAND ----------

dfp = pd.DataFrame(df)

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_path)
model.predict(df)

# COMMAND ----------

# File location and type
file_location = "/FileStore/Sample.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

import mlflow

# COMMAND ----------

# load input data table as a Spark DataFrame
#input_data = spark.table(df)
predict = mlflow.pyfunc.spark_udf(spark, model_path)
#df1 = input_data.withColumn("prediction", model_udf())
df1 = df.withColumn("prediction", predict(struct("fixed acidity"
                                                 ,"volatile acidity"
                                                 ,"citric acid"
                                                 ,"residual sugar"
                                                 ,"chlorides"
                                                 ,"free sulfur dioxide"
                                                 ,"total sulfur dioxide"
                                                 ,"density"
                                                 ,"pH"
                                                 ,"sulphates"
                                                 ,"alcohol"
                                                 )))

# COMMAND ----------

df1.count()

# COMMAND ----------

df1.show(n=400)
