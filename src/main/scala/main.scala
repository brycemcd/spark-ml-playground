package basic

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

class SparkSetup() {
  def sparkPool() = "hi"
}

object Main {
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

  val sc = new SparkContext(conf)



  def matchYesNo(vari: String) : Int = vari match {
    case "yes" => 1
    case "no"  => 0
  }

  def nephritisPoint(dataString: String) : LabeledPoint = {
  //def nephritisPoint(dataString: String) = {
      val data = dataString.split("\t") // this is 2 spaces. This notation is bad
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

  def main(args: Array[String]) = {
    // NOTE: dataset is @ https://archive.ics.uci.edu/ml/datasets/URL+Reputation
    // https://archive.ics.uci.edu/ml/machine-learning-databases/acute/
    val f = sc.textFile("hdfs://spark3.thedevranch.net/classifications-datasets/diagnosis.data.unix")
    val nephritis = f.map{ row =>
      nephritisPoint(row)
    }

    //val data = MLUtils.loadLibSVMFile(sc, "hdfs://spark3.thedevranch.net/classifications-datasets/diagnosis.data")
    val splits = nephritis.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0)//.cache()
    val test = splits(1) //.cache()

    //println(training.take(1).foreach(println))

    // this syntax is not great. It indicates that index 1, has 2 categories
    val categoricalFeaturesInfo = Map[Int, Int]((1 -> 2), (2 -> 2))
    // FIXME - make sure categorical vars are used
    println("===== TRAINING =====")
    println("data set count: " + training.count)

    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)

    println("===== EVAL =====")
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    predictionAndLabels.take(5).foreach(println)

    println("===== MODEL PERF =====")
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    //println("Precision = " + metrics.precision)
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    println("[Model Perf] ROC: " + metrics.roc)
    println("[Model Perf] AUC: " + metrics.areaUnderROC)

    sc.stop()
  }
}
