import pandas as pd
import numpy as np
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
import random as rnd
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import  cross_val_score
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

path="/FileStore/tables/p0gmma661494630018740/new1.csv"
data = sqlContext.read.format("com.databricks.spark.csv").option("header",True).option("inferSchema", True).option("delimiter",",").load(path) 

labelIndexer = StringIndexer(inputCol="Survived", outputCol="ind").fit(data)
assembler = VectorAssembler(
    inputCols=["Pclass","Sex","Age","SibSp","Parch","Ticket","Cabin","Embarked"],outputCol="features")

d=1
while(d<2): 
  (trainingData, testData) = data.randomSplit([0.8, 0.2],seed=11L)
  dt = DecisionTreeClassifier(labelCol="ind", featuresCol="features")
#   dt.setParams(maxDepth=20, maxBins=150)
  pipeline = Pipeline(stages=[labelIndexer, assembler, dt])

  paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [5,8,10,15])
             .addGrid(dt.maxBins, [5,10,30,50])
             .addGrid(dt.impurity, ["gini","entropy"])
             .build())
  evaluator = MulticlassClassificationEvaluator().setLabelCol("Survivied")

  cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid,evaluator=evaluator,numFolds=10)
  cvmodel=cv.fit(trainingData)

  predictions = cvmodel.transform(testData)
  pred=cvmodel.transform(trainingData)
  predictions.select("prediction", "ind", "features")
  evaluator = MulticlassClassificationEvaluator(
  labelCol="ind", predictionCol="prediction", metricName="accuracy")
  accuracy = evaluator.evaluate(predictions)
  print d
  print("Test Accuracy = %g " % (accuracy))
  # 
  pred.select("prediction", "ind", "features")
  evaluatorr = MulticlassClassificationEvaluator(
  labelCol="ind", predictionCol="prediction", metricName="accuracy")
  accuracy = evaluator.evaluate(pred)
  print("Train Accuracy = %g " % (accuracy))
  d=d+1