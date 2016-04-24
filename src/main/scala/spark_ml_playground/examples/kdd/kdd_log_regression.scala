package spark_ml_playground.examples.kdd

import spark_ml_playground.datasets._
import spark_ml_playground.datamodels._
import spark_ml_playground.modelparams._

import java.util.Calendar

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

object KddLogisticRegression extends DataModel[
  KDD,
  LogisticRegressionModel,
  SGDModelParams
] {

  // TODO: just make this the class name without path
  val persistedModelName : String = "kddlogisticregression"
  private val modelPersistancePath : String = "hdfs://spark3.thedevranch.net/ml-models/"+ persistedModelName


  def allData(sc : SparkContext) = KDD.cachedModelData(sc)

  def train(modelParams : SGDModelParams,
            trainingData: RDD[LabeledPoint]) : LogisticRegressionModel = {

    var model = new LogisticRegressionWithSGD()
      model.optimizer.
      setNumIterations(modelParams.numIterations).
      setRegParam(modelParams.regParam)
    model.run(trainingData)
  }

  def minGenerateModelParams = {
    for(regParam <- (0.00001 to 0.0001 by 0.0005);
        numIterations <- (10 to 200 by 100) ) yield SGDModelParams(regParam, numIterations)
  }

  def maxGenerateModelParams = {
    for(regParam <- (0.00001 to 0.1 by 0.0005);
        numIterations <- (10 to 1100 by 100) ) yield SGDModelParams(regParam, numIterations)
  }

  def persistExploratoryResults(results : RDD[Perf[SGDModelParams]]) = {
    results.saveAsTextFile("hdfs://spark3.thedevranch.net/model_results/"+persistedModelName +"/results-" + Calendar.getInstance().getTimeInMillis)
  }


  def persistModel(sc: SparkContext, model: LogisticRegressionModel) = {
    model.save(sc, modelPersistancePath)
  }

  def loadPersistedModel(sc : SparkContext) : LogisticRegressionModel = {
    LogisticRegressionModel.load(sc, modelPersistancePath)
  }
}
