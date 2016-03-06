package spark_ml_playground

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object ExploreKDD extends Explore[KDD, SVMModelParams] {
  def allData : RDD[LabeledPoint] = KDD.cachedModelData(sc)

  def results = {
    val logSummary = LogisticRegression.exploreTraining(trainingSet, testSet) sortBy (_.wPrecision)
    sc.stop()

    println("=== Worst Model: " + logSummary.head)
    println("=== Best Model: "  + logSummary.last)
    logSummary
  }

  def main(args: Array[String]) = {
    Logger.getRootLogger().setLevel(Level.ERROR)

    println("===== Explore Mode =====")
    ExploreKDD.results
    println("===== Done =====")
  }
}
