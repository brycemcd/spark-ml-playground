//TODO learn how to namespace
package spark_ml_playground

import java.util.Calendar

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.optimization.L1Updater

object KddSVM extends DataModel[
  KDD,
  SVMModel,
  SGDModelParams
] {

  val persistedModelName : String = "kddsvm"
  private val modelPersistancePath : String = "hdfs://spark3.thedevranch.net/ml-models/"+ persistedModelName


  def allData(sc : SparkContext) = KDD.cachedModelData(sc)
  def train(modelParams: SGDModelParams,
            trainingData: RDD[LabeledPoint]) : SVMModel = {

    val model = new SVMWithSGD()
      model.optimizer.
      setNumIterations(modelParams.numIterations).
      setRegParam(modelParams.regParam)
    model.run(trainingData)
  }

  def persistExploratoryResults(results : RDD[Perf[SGDModelParams]]) = {
    results.saveAsTextFile("hdfs://spark3.thedevranch.net/model_results/"+persistedModelName +"/results-" + Calendar.getInstance().getTimeInMillis)
  }

  def persistModel(sc: SparkContext, model: SVMModel) = {
    model.save(sc, modelPersistancePath)
  }

  def loadPersistedModel(sc : SparkContext) : SVMModel = {
    SVMModel.load(sc, modelPersistancePath)
  }
}
