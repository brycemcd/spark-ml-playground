package spark_ml_playground

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

import org.apache.log4j.Logger
import org.apache.log4j.Level

object AcuteInflammation {

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
                              patientUrinePushing,
                              patientMicturitionPain,
                              patientBurning
                            )
      LabeledPoint(outcomeNephritis, vec)
  }

  def train(trainingData: org.apache.spark.rdd.RDD[LabeledPoint]) : LogisticRegressionModel = {
    new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
    //new LogisticRegressionWithSGD()
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
    println("[Model Perf] Confusion Matrix: ")

    val multMetrics = new MulticlassMetrics(predictionAndLabels)
    println(multMetrics.confusionMatrix)
    metrics
  }

  def trainTestAndEval(sc : SparkContext) = {

    // NOTE: dataset is @ https://archive.ics.uci.edu/ml/datasets/URL+Reputation
    //val data = MLUtils.loadLibSVMFile(sc, "hdfs://spark3.thedevranch.net/classifications-datasets/diagnosis.data")

    // https://archive.ics.uci.edu/ml/machine-learning-databases/acute/
    val diagnosisData = sc.textFile("hdfs://spark3.thedevranch.net/classifications-datasets/diagnosis.data.unix")
    val nephritis = diagnosisData.map(row => nephritisPoint(row) )
    val splits = nephritis.randomSplit(Array(0.8, 0.2))
    val training = splits(0)//.cache()
    val test = splits(1) //.cache()

    println("===== TRAINING =====")
    println("data set count: " + training.count)
    val model = train(training)


    // this syntax is not great. It indicates that index 1, has 2 categories
    // FIXME - make sure categorical vars are used
    val categoricalFeaturesInfo = Map[Int, Int]((1 -> 2), (2 -> 2))


    println("===== PREDICT =====")
    println("weights: " + model.weights)
    val predictionAndLabels = predict(model, test)

    // output some cases to see how well we did
    //predictionAndLabels.take(5).foreach(println)

    println("===== MODEL PERF =====")
    modelPerformance(predictionAndLabels)
  }
}
