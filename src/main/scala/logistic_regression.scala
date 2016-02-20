package spark_ml_playground

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

import org.apache.log4j.Logger
import org.apache.log4j.Level

class SparkSetup {
  // NOTE: change this to use the cluster later
  val conf = new SparkConf()
  .setAppName("classificationTest")
  .setMaster("local[6]")
  //.set("spark.executor.memory", "30g")
  //.set("spark.executor-memory", "30g")
  //.set("spark.driver.memory", "30g")
  //.set("spark.driver-memory", "30g")
  //.set("spark.storage.memoryFraction", "0.9999")
  //.set("spark.eventLog.enabled", "true")
  //.set("spark.eventLog.dir", "/home/brycemcd/Downloads")

  def sc = {
    // FIXME: is this the only way to shut the logger up?
    val s = new SparkContext(conf)
    Logger.getRootLogger().setLevel(Level.ERROR)
    s
  }
}

object AcuteInflammation {

  val sc = new SparkSetup().sc

  def matchYesNo(vari: String) : Int = vari match {
    case "yes" => 1
    case "no"  => 0
  }

  def nephritisPoint(dataString: String) : LabeledPoint = {
      val data = dataString.split("\t")
      val patientTemp = data(0).replaceAll(",", ".").toDouble
      val patientNausea = matchYesNo( data(1) )
      val patientLumbarPain = matchYesNo( data(2) )
      val patientUrinePushing = matchYesNo( data(3) )
      val patientMicturitionPain = matchYesNo( data(4) )
      val patientBurning = matchYesNo( data(5) )
      val outcomeInflamation = matchYesNo( data(6) )
      val outcomeNephritis  = matchYesNo( data(7) )

      val vec = Vectors.dense(patientTemp,
                              patientNausea,
                              patientLumbarPain,
                              patientUrinePushing

                            )
      LabeledPoint(outcomeNephritis, vec)
  }

  // https://archive.ics.uci.edu/ml/machine-learning-databases/acute/
  val diagnosisData = sc.textFile("hdfs://spark3.thedevranch.net/classifications-datasets/diagnosis.data.unix")

  val nephritis = diagnosisData.map(row => nephritisPoint(row) )

  def train(trainingData: org.apache.spark.rdd.RDD[LabeledPoint]) : LogisticRegressionModel = {
    new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(trainingData)
  }

  def predict(model : LogisticRegressionModel,
    data : org.apache.spark.rdd.RDD[LabeledPoint]) = {

    data.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
  }

  def modelPerformance(predictionAndLabels: org.apache.spark.rdd.RDD[(Double, Double)]) = {
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    println("[Model Perf] AUC: " + metrics.areaUnderROC)
    metrics
  }

  def trainTestAndEval = {

    // NOTE: dataset is @ https://archive.ics.uci.edu/ml/datasets/URL+Reputation
    //val data = MLUtils.loadLibSVMFile(sc, "hdfs://spark3.thedevranch.net/classifications-datasets/diagnosis.data")

    val splits = nephritis.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0)//.cache()
    val test = splits(1) //.cache()

    println("===== TRAINING =====")
    println("data set count: " + training.count)
    val model = train(training)


    // this syntax is not great. It indicates that index 1, has 2 categories
    // FIXME - make sure categorical vars are used
    val categoricalFeaturesInfo = Map[Int, Int]((1 -> 2), (2 -> 2))


    println("===== PREDICT =====")
    val predictionAndLabels = predict(model, test)

    // output some cases to see how well we did
    predictionAndLabels.take(5).foreach(println)

    println("===== MODEL PERF =====")
    modelPerformance(predictionAndLabels)


    // NOTE: put this somewhere better
    sc.stop()
  }
}
