package spark_ml_playground

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics

class SparkSetup(
  val host : String = "local[4]"
) {
  val conf = {
    println("==== host " + host)
    new SparkConf()
    .setAppName("ML Playground")
    //.setMaster(host)
  }
  //.set("spark.executor.memory", "30g")
  //.set("spark.executor-memory", "30g")
  //.set("spark.driver.memory", "30g")
  //.set("spark.driver-memory", "30g")
  //.set("spark.storage.memoryFraction", "0.9999")
  //.set("spark.eventLog.enabled", "true")

  def sc = {
    // FIXME: is this the only way to shut the logger up?
    val s = new SparkContext(conf)
    Logger.getRootLogger().setLevel(Level.ERROR)
    s
  }
}

object Explore {
  def main(args: Array[String]) = {
    Logger.getRootLogger().setLevel(Level.ERROR)
    val host = "local[*]"
    //val host = "spark://10.1.2.244:7077"

    val sc = new SparkSetup(host).sc
    sc.setLogLevel("ERROR")
    //val preparedData : RDD[LabeledPoint] = KDD.prepareData(sc)
    val preparedData : RDD[LabeledPoint] = KDD.cachedModelData(sc)

    val splits = preparedData
      .randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1).cache()

    println("===== Explore Mode =====")

    //val svmSummary = SVM.exploreTraining(training, test) sortBy (_.wPrecision)
    //println("=== Worst Model: " + svmSummary.head)
    //println("=== Best Model: "  + svmSummary.last)

    val logSummary = LogisticRegression.exploreTraining(training, test) sortBy (_.wPrecision)
    println("=== Worst Model: " + logSummary.head)
    println("=== Best Model: "  + logSummary.last)

    sc.stop()
  }
}

object Main {

  def mainOLD(args: Array[String]) = {
    Logger.getRootLogger().setLevel(Level.WARN)
    val host = "local[6]"
    //val host = "spark://10.1.2.244:7077"

    val sc = new SparkSetup(host).sc
    sc.setLogLevel("ERROR")
    //val preparedData : RDD[LabeledPoint] = KDD.prepareData(sc)
    val preparedData : RDD[LabeledPoint] = KDD.cachedModelData(sc)

    val splits = preparedData
      .randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    println("===== TRAINING =====")
    println("data set count: " + training.count)
    //val model = LogisticRegression.train(training, 2)
    val model = SVM.train(training, SVMModelParams(1.0, 100))

    println("===== Evaluation =====")
    println("weights: " + model.weights)
    //val predictionAndLabels = LogisticRegression.predict(model, test)
    val evaluation = SVM.evaluateModel(model, test)

    println("===== MODEL PERF =====")
    //LogisticRegression.modelPerformance(predictionAndLabels)
    //modelPerformance(evaluation)

    sc.stop()
  }
}
