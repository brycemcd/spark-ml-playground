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

import org.apache.spark.rdd.RDD

object Predict {
  def main(args: Array[String]) = {
    val sc = new SparkSetup().sc
    val preparedData : RDD[LabeledPoint] = KDD.cachedModelData(sc)


    for(i <- (0 to 10)) {
      val dataPoint = preparedData.takeSample(false, 1)(0)
      val prediction = LogisticRegression.predict(sc,
                                            "kdd-logisticregression",
                                            dataPoint.features)

      //println("data: "+ dataPoint )
      println("---")
      println("prediction: " + prediction)
      println("label: " + dataPoint.label)
      (dataPoint.label.toString == prediction.toString)
    } //.map { result => println(result) }

    sc.stop()
  }
}
