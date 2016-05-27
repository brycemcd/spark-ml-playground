package spark_ml_playground.examples.prosper

import spark_ml_playground._
import spark_ml_playground.datasets._
import spark_ml_playground.datamodels._
import spark_ml_playground.modelparams._

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

object LogRegression extends DataModel[
  ProsperData,
  LogisticRegressionModel,
  SGDModelParams
] {
  val persistedModelName : String = "prosperlogisticregression"
  private val modelPersistancePath : String = "hdfs://spark3.thedevranch.net/ml-models/"+ persistedModelName

  def allData(sc : SparkContext) = ProsperData.cachedModelData(sc)

  def train(modelParams : SGDModelParams,
            trainingData: RDD[LabeledPoint]) : LogisticRegressionModel = {

    var model = new LogisticRegressionWithSGD()
      //model.setIntercept(true)

      model.optimizer.
      setNumIterations(modelParams.numIterations).
      setRegParam(modelParams.regParam)
    model.run(trainingData)
  }

  def minGenerateModelParams = {
    for(regParam <- (30 to 40 by 10);
        numIterations <- (100 to 200 by 100) ) yield SGDModelParams(regParam, numIterations)
  }

  override def exploreTraining(sc : SparkContext,
                      modelParams: Seq[SGDModelParams],
                      trainingData: RDD[LabeledPoint],
                      testData: RDD[LabeledPoint],
                      tx: (SGDModelParams, RDD[LabeledPoint]) => LogisticRegressionModel
                    ) : RDD[Perf[SGDModelParams]] = {


    // 1. Generate model params
    // 2. Develop models on each param set
    println("=== override training "+ modelParams.length + " models")
    //val results = modelParams.par.map { modelP =>
    val results = modelParams.map { modelP =>
      val model = tx(modelP, trainingData)
      (modelP, model)
    }.map { case(modelP, model) =>
      // 3. test model
      println("===")
      //model.setThreshold(0.50)
      println(model.getThreshold)
      println(model.intercept)
      println(model.weights)
      println("===")
      val predictionLabel = GLMModelDevelopment.evaluateModel[LogisticRegressionModel](model, testData)
      (predictionLabel, modelP)
    }.map { case(modelPredictions, modelP) =>
      // 4. Collect model evaluation metrics
      val metrics = new MulticlassMetrics(modelPredictions)

      println("---")
      metrics.labels.foreach(println)
      println(metrics.confusionMatrix)
      println("---")

      Perf[SGDModelParams](modelP, metrics.weightedRecall, metrics.weightedPrecision, metrics.recall)
    }.toList

    val rddResults = sc.parallelize(results)
    rddResults
  }

  def main(args: Array[String]) = {
    lazy val sc : SparkContext = new SparkSetup().sc
    val data = ProsperData.prepareRawData(sc)
    //data.take(10).foreach(println)
    val minParams = minGenerateModelParams

    // training step completely hangs
    val trs_ = trainingSet(sc)
    val tes_ = testSet(sc)

    // required?!?
    trs_.count
    tes_.count

    val logResults = exploreTraining(sc, minParams, trs_, tes_, LogRegression.train)
    logResults.foreach(println)
    sc.stop()
  }
}
