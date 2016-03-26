package spark_ml_playground

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics


object Main {
  def main(args: Array[String]) = {
    //implicit val sc : SparkContext = new SparkSetup().sc
    //lazy val sc : SparkContext = new SparkSetup().sc
    // NOTE: this is a sanity check that I've implmented all the methods
    // necessary for a data model
    println("===== Explore Mode =====")
    //val allModelParams = KddLogisticRegression.exploreTrainingResults
    val allModelParams = KddLogisticRegression.everythingInOne

    println("===== Persist Best Model =====")
    //val bestModelParams = allModelParams.last.modelParams
    // NOTE: works, but throws exception if it already exists
    //KddLogisticRegression.persistModel(bestModel)

    println("===== Train Model With Arbitrary Params =====")
    //val bestModel = KddLogisticRegression.train(bestModelParams)

    println("===== Load Persisted Model =====")
    //val bestModel = KddLogisticRegression.loadPersistedModel

    //println("===== Predict  =====")
    //for(i <- (0 to 10)) {
      //val dataPoint = KddLogisticRegression.testSet.takeSample(false, 1)(0)
      //val prediction = KddLogisticRegression.predict(dataPoint.features)
      //println("---")
      //println("prediction: " + prediction.toString)
      //println("label: " + dataPoint.label)
      //(dataPoint.label.toString == prediction.toString)
    //}
    println("===== Done =====")
    KddLogisticRegression.sc.stop()
  }
}
