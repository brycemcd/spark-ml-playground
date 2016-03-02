package spark_ml_playground

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint

import scala.reflect.ClassTag

import org.apache.spark.mllib.regression.GeneralizedLinearModel

abstract class ModelParams

/*
* M = Linear model generalization 
*/
abstract class Model[
  M <: GeneralizedLinearModel,
  P <: ModelParams
  ] {
  def train(trainingData: org.apache.spark.rdd.RDD[LabeledPoint], modelParams: P) : M

  def evaluateModel(model : M,
                    data : org.apache.spark.rdd.RDD[LabeledPoint]) : MulticlassMetrics

  def exploreTraining(trainingData: org.apache.spark.rdd.RDD[LabeledPoint],
                      testData: org.apache.spark.rdd.RDD[LabeledPoint])
  // TODO: add predict method
}

case class Perf[P <: ModelParams](modelParams : P, wRecall: Double, wPrecision: Double)

//trait PerformanceSummary(perfSummaries: Seq[Perf[Any]]) {

  //def sortSummaries = (perfSummaries sortBy (_.wPrecision))

  //def bestModel = modelPerfToString(sortSummaries.last)
  //def worstModel = modelPerfToString(sortSummaries.head)

  //def bestModelParam = sortSummaries.last.modelParams

  //private def modelPerfToString(perf: Perf[Any]) =  perf.modelParams.toString + " resulted in weighted recall " + perf.wRecall + " and weighted Precision " + perf.wPrecision

  //def printSummary = {
    //sortSummaries.foreach{ x =>
      //println(x)
    //}
  //}
//}
