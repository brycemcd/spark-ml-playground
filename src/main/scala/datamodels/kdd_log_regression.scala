package spark_ml_playground

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vector
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


  def allData = KDD.cachedModelData(sc)

  def train(modelParams : SGDModelParams) : LogisticRegressionModel = {
    var model = new LogisticRegressionWithSGD()
      model.optimizer.
      setNumIterations(modelParams.numIterations).
      setRegParam(modelParams.regParam)
    model.run(trainingSet)
  }

  private def generateModelParams : Seq[SGDModelParams] = {
    //for(regParam <- (0.00001 to 1.00 by 0.0005);
      //numIterations <- (100 to 3000 by 300) ) yield SGDModelParams(regParam, numIterations)
    for(regParam <- (0.00001 to 0.0001 by 0.0005);
      numIterations <- (100 to 200 by 100) ) yield SGDModelParams(regParam, numIterations)
  }

  def exploreTraining(trainingData: RDD[LabeledPoint],
                      testData: RDD[LabeledPoint]) : Seq[Perf[SGDModelParams]] = {

    generateModelParams.map { modelParam =>

      val model = this.train(modelParam)
      val metrics = this.evaluateModel(model, testData)

      //FIXME: this is for debugging
      println(modelParam.regParam + "," + modelParam.numIterations + "," + metrics.weightedRecall + "," + metrics.weightedPrecision)
      Perf[SGDModelParams](modelParam, metrics.weightedRecall, metrics.weightedPrecision)
    }
  }

  def exploreTrainingResults = {
    val logSummary = exploreTraining(trainingSet, testSet) sortBy (_.wPrecision)

    println("=== Worst Model: " + logSummary.head)
    println("=== Best Model: "  + logSummary.last)
    logSummary
  }

  def persistModel(model: LogisticRegressionModel) = {
    model.save(sc, modelPersistancePath)
  }

  def loadPersistedModel : LogisticRegressionModel = {
    LogisticRegressionModel.load(sc, modelPersistancePath)
  }

  def predict(features: Vector, model: LogisticRegressionModel = bestModel) : Double = {
    model.predict(features)
  }

  def evaluateModel(model : LogisticRegressionModel,
                    data : RDD[LabeledPoint]) = {

    val predictionAndLabels = data.map { case LabeledPoint(label, features) =>
      val prediction = predict(features, model)
      (prediction, label)
    }
    new MulticlassMetrics(predictionAndLabels)
  }
}
