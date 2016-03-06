package spark_ml_playground

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint

import scala.reflect.ClassTag

import org.apache.spark.mllib.regression.GeneralizedLinearModel
import org.apache.spark.rdd.RDD

abstract class ModelParams

trait Model[
  M <: GeneralizedLinearModel,
  P <: ModelParams
  ] {
  def train(trainingData: RDD[LabeledPoint], modelParams: P) : M

  def evaluateModel(model : M,
                    data :RDD[LabeledPoint]) : MulticlassMetrics

  def exploreTraining(trainingData:RDD[LabeledPoint],
                      testData: RDD[LabeledPoint]) : Seq[Perf[P]]

  // TODO: add predict method
}

// TODO: Really need to get this P out of here. It affects Model + Explore
case class Perf[P <: ModelParams](modelParams : P, wRecall: Double, wPrecision: Double)
