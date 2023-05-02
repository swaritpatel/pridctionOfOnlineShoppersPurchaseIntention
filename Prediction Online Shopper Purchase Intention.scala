// Databricks notebook source
// MAGIC %md
// MAGIC
// MAGIC # Prediction of Online Shoppers purchasing intention Project using Apache Spark Machine Learning

// COMMAND ----------

// MAGIC %md ### Loading Dataset

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC val onlineshopperDF = sqlContext.read.format("csv").option("header", "true").option("inferSchema", "true").option("delimiter", ",").load("/FileStore/tables/online_shoppers_intention.csv")
// MAGIC
// MAGIC display(onlineshopperDF)

// COMMAND ----------

// DBTITLE 1,Printing Schema of Dataframe
// MAGIC %scala
// MAGIC
// MAGIC onlineshopperDF.printSchema();

// COMMAND ----------

// DBTITLE 1,Finding count, mean, maximum, standard deviation and minimum
// MAGIC %scala
// MAGIC
// MAGIC display(onlineshopperDF.select("Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend", "Revenue").describe())

// COMMAND ----------

// DBTITLE 1,Creating Temp View from Dataframe 
// MAGIC %scala
// MAGIC
// MAGIC onlineshopperDF.createOrReplaceTempView("WebData")

// COMMAND ----------

// DBTITLE 1,Querying the Temporary View
// MAGIC %sql
// MAGIC
// MAGIC select * from WebData;

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC #Exploratory Data Analysis

// COMMAND ----------

// MAGIC %md
// MAGIC ###Distribution of our Labels:

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select Revenue as CustomerboughtProduct, count(Revenue) as counts  from WebData group by Revenue;

// COMMAND ----------

// DBTITLE 1,Displaying Percentage of Customer who bought Product (True/False)
// MAGIC %sql
// MAGIC
// MAGIC select Revenue as Customerwillbuy, count(Revenue) as counts  from WebData group by Revenue;

// COMMAND ----------

// DBTITLE 1,Scatter Plots to All
// MAGIC %sql
// MAGIC
// MAGIC select Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay, Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend, Revenue from WebData;

// COMMAND ----------

// DBTITLE 1,Purchase on Weekends
// MAGIC %sql
// MAGIC
// MAGIC select Weekend, Count(Weekend) from WebData group by Weekend;

// COMMAND ----------

// DBTITLE 1,Types of Visitors
// MAGIC %sql
// MAGIC
// MAGIC select VisitorType, count(VisitorType) from WebData group by VisitorType

// COMMAND ----------

// DBTITLE 1,Types of Browser
// MAGIC %sql
// MAGIC
// MAGIC select Browser, Count(Browser) as BrowserType from WebData group by Browser;

// COMMAND ----------

// DBTITLE 1,Types of Traffic
// MAGIC %sql
// MAGIC
// MAGIC select TrafficType, count(TrafficType) from WebData group by TrafficType order by TrafficType;

// COMMAND ----------

// DBTITLE 1,Regions
// MAGIC %sql
// MAGIC
// MAGIC select Region, count(Region) from WebData group by Region;

// COMMAND ----------

// DBTITLE 1,Types of Operating Systems
// MAGIC %sql
// MAGIC
// MAGIC select OperatingSystems, count(OperatingSystems) from WebData group by OperatingSystems;

// COMMAND ----------

// DBTITLE 1,Months
// MAGIC %sql
// MAGIC
// MAGIC select Month, count(Month) from WebData group by Month;

// COMMAND ----------

// DBTITLE 1,Informational Duration VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC select Informational_Duration, Revenue from Webdata group by Informational_Duration,Revenue;

// COMMAND ----------

// DBTITLE 1,Administrative Duration VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC select Administrative_Duration,Revenue from Webdata group by Administrative_Duration,Revenue;

// COMMAND ----------

// DBTITLE 1,ProductRelated Duration VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC select ProductRelated_Duration,Revenue from Webdata group by ProductRelated_Duration,Revenue;

// COMMAND ----------

// DBTITLE 1,Exit Rates VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC select ExitRates,Revenue from Webdata group by ExitRates,Revenue;

// COMMAND ----------

// DBTITLE 1,Page Values VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC
// MAGIC select PageValues,Revenue from Webdata group by PageValues,Revenue;

// COMMAND ----------

// DBTITLE 1,Bounce Rates VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC
// MAGIC select BounceRates,Revenue from Webdata group by BounceRates,Revenue;

// COMMAND ----------

// DBTITLE 1,Type of Traffic VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC
// MAGIC select TrafficType, count(TrafficType), Revenue from Webdata group by TrafficType,Revenue;

// COMMAND ----------

// DBTITLE 1,Visitor Type VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC
// MAGIC select VisitorType, count(VisitorType), Revenue from Webdata group by VisitorType,Revenue;

// COMMAND ----------

// DBTITLE 1,Region Type VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC
// MAGIC select Region, count(Region), Revenue from Webdata group by Region,Revenue;

// COMMAND ----------

// DBTITLE 1,Administrative VS Informational
// MAGIC %sql
// MAGIC
// MAGIC select Administrative, Informational from WebData;

// COMMAND ----------

// MAGIC %md ## Implementing a Logistic Regression Model

// COMMAND ----------

// DBTITLE 1,Importing Apache Spark Library
// MAGIC %scala
// MAGIC
// MAGIC import org.apache.spark.sql.functions._
// MAGIC import org.apache.spark.sql.Row
// MAGIC import org.apache.spark.sql.types._
// MAGIC
// MAGIC import org.apache.spark.ml.classification.LogisticRegression
// MAGIC import org.apache.spark.ml.feature.VectorAssembler

// COMMAND ----------

// MAGIC %md ### Prepare the Training Data

// COMMAND ----------

// MAGIC %md ###VectorAssembler()

// COMMAND ----------

// DBTITLE 1,List all String Data Type Columns in an Array in further processing
// MAGIC %scala
// MAGIC
// MAGIC var StringfeatureCol = Array("Month", "VisitorType")

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC ###StringIndexer

// COMMAND ----------

// DBTITLE 1,Example of StringIndexer
import org.apache.spark.ml.feature.StringIndexer

val df = spark.createDataFrame(
  Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
).toDF("id", "category")

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")

val indexed = indexer.fit(df).transform(df)

display(indexed)

// COMMAND ----------

// MAGIC %md ### Define the Pipeline

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC import org.apache.spark.ml.attribute.Attribute
// MAGIC import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
// MAGIC import org.apache.spark.ml.{Pipeline, PipelineModel}
// MAGIC
// MAGIC val indexers = StringfeatureCol.map { colName =>
// MAGIC   new StringIndexer().setInputCol(colName).setOutputCol(colName + "_indexed")
// MAGIC }
// MAGIC
// MAGIC val pipeline = new Pipeline()
// MAGIC                     .setStages(indexers)      
// MAGIC
// MAGIC val ShopperDF = pipeline.fit(onlineshopperDF).transform(onlineshopperDF)

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC ShopperDF.printSchema()

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC display(ShopperDF)

// COMMAND ----------

// DBTITLE 1,Converting Boolean Value to Integer since Model cannot process Boolean Value
// MAGIC %scala
// MAGIC
// MAGIC val FinalShopperDF = ShopperDF
// MAGIC   .withColumn("RevenueInt",$"Revenue".cast("Int"))
// MAGIC   .withColumn("WeekendInt",$"Weekend".cast("Int"))

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC display(FinalShopperDF)

// COMMAND ----------

// MAGIC %md ### Split the Data

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC val splits = FinalShopperDF.randomSplit(Array(0.7, 0.3))
// MAGIC val train = splits(0)
// MAGIC val test = splits(1)
// MAGIC val train_rows = train.count()
// MAGIC val test_rows = test.count()
// MAGIC println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

// COMMAND ----------

// DBTITLE 1,VectorAssembler() that combines categorical features into a single vector
// MAGIC %scala
// MAGIC
// MAGIC val assembler = new VectorAssembler().setInputCols(Array("Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType_indexed", "WeekendInt", "RevenueInt")).setOutputCol("features")
// MAGIC
// MAGIC val training = assembler.transform(train).select($"features", $"RevenueInt".alias("label"))
// MAGIC
// MAGIC training.show(false)

// COMMAND ----------

// MAGIC %md ### Train a Regression Model

// COMMAND ----------

// MAGIC %scala
// MAGIC import org.apache.spark.ml.classification.LogisticRegression
// MAGIC
// MAGIC val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3)
// MAGIC val model = lr.fit(training)
// MAGIC println("Model Trained!")

// COMMAND ----------

// MAGIC %md ### Prepare the Testing Data

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC val testing = assembler.transform(test).select($"features", $"RevenueInt".alias("trueLabel"))
// MAGIC testing.show(false)

// COMMAND ----------

// MAGIC %md ### Test the Model

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC val prediction = model.transform(testing)
// MAGIC val predicted = prediction.select("features", "prediction", "trueLabel")
// MAGIC predicted.show()

// COMMAND ----------

// MAGIC %md ### Computing Confusion Matrix Metrics

// COMMAND ----------

// MAGIC %scala
// MAGIC
// MAGIC val tp = predicted.filter("prediction == 1 AND truelabel == 1").count().toFloat
// MAGIC val fp = predicted.filter("prediction == 1 AND truelabel == 0").count().toFloat
// MAGIC val tn = predicted.filter("prediction == 0 AND truelabel == 0").count().toFloat
// MAGIC val fn = predicted.filter("prediction == 0 AND truelabel == 1").count().toFloat
// MAGIC   val metrics = spark.createDataFrame(Seq(
// MAGIC  ("TP", tp),
// MAGIC  ("FP", fp),
// MAGIC  ("TN", tn),
// MAGIC  ("FN", fn),
// MAGIC  ("Precision", tp / (tp + fp)),
// MAGIC  ("Recall", tp / (tp + fn)))).toDF("metric", "value")
// MAGIC metrics.show()

// COMMAND ----------

// MAGIC %md ### Classification model Evaluation

// COMMAND ----------

// MAGIC %scala
// MAGIC import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// MAGIC
// MAGIC val evaluator = new MulticlassClassificationEvaluator()
// MAGIC   .setLabelCol("trueLabel")
// MAGIC   .setPredictionCol("prediction")
// MAGIC   .setMetricName("accuracy")
// MAGIC val accuracy = evaluator.evaluate(prediction)

// COMMAND ----------

// MAGIC %md ##Accuracy of the Model
// MAGIC ####Accuracy: 0.9208535926526202
