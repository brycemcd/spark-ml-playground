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
    lazy val sc : SparkContext = new SparkSetup().sc
    // NOTE: this is a sanity check that I've implmented all the methods
    // necessary for a data model
    println("===== Explore Mode =====")
    //val allModelParams = KddLogisticRegression.exploreTrainingResults

    val minParams = KddLogisticRegression.minGenerateModelParams

    // training step completely hangs
    val trs_ = KddLogisticRegression.trainingSet(sc)
    val tes_ = KddLogisticRegression.testSet(sc)
    // required?!?
    trs_.count
    tes_.count

    val bcts = sc.broadcast(trs_).value.cache()
    val bcss = sc.broadcast(tes_).value.cache()

    bcts.count
    bcss.count

    val logResults = KddLogisticRegression.exploreTraining(sc, minParams, bcts, bcss)
    KddLogisticRegression.persistExploratoryResults(logResults)

    val svmResults = KddSVM.exploreTraining(sc, minParams, bcts, bcss)
    KddSVM.persistExploratoryResults(svmResults)

    println("===== Done =====")
    sc.stop()
  }
}
