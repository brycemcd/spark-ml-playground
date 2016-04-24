package spark_ml_playground.datamodels

import spark_ml_playground.datasets.DataSet
import spark_ml_playground.modelparams._
// FIXME: not all of these are needed
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.GeneralizedLinearModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import scala.reflect.ClassTag

// NOTE: this does too much. Concerns of model training, model storage + retreival
// prediction and exploration should be separated

trait ModelDevelopment {
  def predict[M <: GeneralizedLinearModel](features: Vector, model: M) : Double
  def evaluateModel[M <: GeneralizedLinearModel](model : M,
                    data : RDD[LabeledPoint]) : RDD[(Double, Double)]
}

object GLMModelDevelopment extends ModelDevelopment {
  def predict[M <: GeneralizedLinearModel](features: Vector, model: M) : Double = {
    model.predict(features)
  }

  def evaluateModel[M <: GeneralizedLinearModel](model : M,
                      data : RDD[LabeledPoint]) : RDD[(Double, Double)] = {

    data.map { case LabeledPoint(label, features) =>
      val prediction = predict[M](features, model)
      (prediction, label)
    }
  }
}

trait DataModel[
  D <: DataSet,
  M <: GeneralizedLinearModel,
  P <: ModelParams
] {
  // DATA

  // TODO: make this a random # or remove it completely
  lazy val seedVal : Long = 11L
  lazy val trainingSetRatio : Double = 0.8
  lazy val testSetRatio : Double = 1.0 - trainingSetRatio
  val persistedModelName : String

  def allData(sc : SparkContext) : RDD[LabeledPoint]
  def splits(sc : SparkContext) = allData(sc).randomSplit(Array(trainingSetRatio, testSetRatio), seed = seedVal)
  def trainingSet(sc : SparkContext) : RDD[LabeledPoint] ={
    val training = splits(sc)(0).cache()
    training
  }

  def testSet(sc : SparkContext) : RDD[LabeledPoint] = splits(sc)(1).cache()


  // Model Development

  // TODO: print top n results?
  // NOTE: use this to test out a bunch of different training parameters
  // and return a list of the parameters with performance metrics
  def exploreTraining(sc : SparkContext,
                      modelParams: Seq[P],
                      trainingData: RDD[LabeledPoint],
                      testData: RDD[LabeledPoint]) : RDD[Perf[P]] = {


    // 1. Generate model params
    // 2. Develop models on each param set
    println("=== training "+ modelParams.length + " models")
    val results = modelParams.par.map { modelP =>
      val model = train(modelP, trainingData)
      (modelP, model)
    }.map { case(modelP, model) =>
      // 3. test model
      val predictionLabel = GLMModelDevelopment.evaluateModel[M](model, testData)
      (predictionLabel, modelP)
    }.map { case(modelPredictions, modelP) =>
      // 4. Collect model evaluation metrics
      val metrics = new MulticlassMetrics(modelPredictions)

      Perf[P](modelP, metrics.weightedRecall, metrics.weightedPrecision)
    }.toList

    val rddResults = sc.parallelize(results)
    rddResults
  }

  // TODO: move this up into GLMModelDevelopment
  def train(modelParams: P, trainingData: RDD[LabeledPoint]) : M



  // Model Use
  //lazy val bestModel : M = loadPersistedModel
  def persistModel(sc : SparkContext, model: M)
  def loadPersistedModel(sc : SparkContext) : M
}
