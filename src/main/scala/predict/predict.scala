package spark_ml_playground

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vector

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD}

/*
* This is meant to marry a spcific model with a specific dataset.
* An individual data set can be modeled different ways by invoking different
* model types
*
* D = data set
* M = model. Should inherit from Model[ M <: GeneralizedLinearModel, P <: ModelParams] (can't get that to work atm)
*/
import org.apache.spark.mllib.regression.GeneralizedLinearModel
trait ModelData[
  D <: DataSource,
  M
] {
  val persistedModelName : String
}
/*
* M = Model Type like LogisticRegression
*/
trait Predict {

  def predict(sc: SparkContext,
              name: String,
              features: Vector) = {

    val model = LogisticRegressionModel.load(sc, s"hdfs://spark3.thedevranch.net/ml-models/$name")
    model.predict(features)
  }
}

object PredictKDD extends Predict with ModelData[KDD, LogisticRegression] {
  val persistedModelName = "kdd-logisticregression"

  def main(args: Array[String]) = {
    val sc = new SparkSetup().sc
    val preparedData : RDD[LabeledPoint] = KDD.cachedModelData(sc)


    for(i <- (0 to 10)) {
      val dataPoint = preparedData.takeSample(false, 1)(0)
      val prediction = predict( sc,
                                persistedModelName,
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
