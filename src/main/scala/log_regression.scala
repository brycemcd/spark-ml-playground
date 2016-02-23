package spark_ml_playground

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import org.apache.spark.mllib.regression.LabeledPoint

object LogisticRegression {

  def train(trainingData: org.apache.spark.rdd.RDD[LabeledPoint], numClasses : Int) : LogisticRegressionModel = {
    new LogisticRegressionWithLBFGS()
      .setNumClasses(numClasses)
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
    multMetrics.labels.foreach(label =>
      println(s"[Model Perf] $label precision:  ${multMetrics.precision(label)}"))
    println(s"[Model Perf] weighted precision: ${multMetrics.weightedPrecision}")
    metrics
  }
}
