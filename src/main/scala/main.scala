package spark_ml_playground

import spark_ml_playground.AcuteInflammation
import spark_ml_playground.KDD
import spark_ml_playground.LogisticRegression

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class SparkSetup(
  val host : String = "local[4]"
) {
  val conf = {
    println("==== host " + host)
    new SparkConf()
    .setAppName("ML Playground")
    .setMaster(host)
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

object Main {

  def main(args: Array[String]) = {
    Logger.getRootLogger().setLevel(Level.WARN)
    val host = "local[6]"
    //val host = "spark://10.1.2.244:7077"

    val sc = new SparkSetup(host).sc
    sc.setLogLevel("ERROR")

    val preparedData : RDD[LabeledPoint] = KDD.prepareData(sc)
    //val preparedData : String = KDD.prepareData(sc)

    val splits = preparedData
      .randomSplit(Array(0.8, 0.2), seed = 11L)

    val training = splits(0).cache()
    val test = splits(1)

    println("===== TRAINING =====")
    println("data set count: " + training.count)
    val model = LogisticRegression.train(training, 2)

    println("===== PREDICT =====")
    println("weights: " + model.weights)
    val predictionAndLabels = LogisticRegression.predict(model, test)

    println("===== MODEL PERF =====")
    LogisticRegression.modelPerformance(predictionAndLabels)

    sc.stop()
  }
}
