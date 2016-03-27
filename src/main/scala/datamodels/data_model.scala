package spark_ml_playground

// FIXME: not all of these are needed
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.GeneralizedLinearModel
import org.apache.spark.mllib.regression.GeneralizedLinearModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import scala.reflect.ClassTag

// NOTE: this does too much. Concerns of model training, model storage + retreival
// prediction and exploration should be separated

trait DataModel[
  D <: DataSet,
  M <: GeneralizedLinearModel,
  P <: ModelParams
] {
  // TODO: make this a random # or remove it completely
  lazy val seedVal : Long = 11L
  lazy val trainingSetRatio : Double = 0.8
  lazy val testSetRatio : Double = 1.0 - trainingSetRatio
  //lazy val bestModel : M = loadPersistedModel

  val persistedModelName : String

  def allData(sc : SparkContext) : RDD[LabeledPoint]
  def splits(sc : SparkContext) = allData(sc).randomSplit(Array(trainingSetRatio, testSetRatio), seed = seedVal)
  def trainingSet(sc : SparkContext) : RDD[LabeledPoint] ={
    val training = splits(sc)(0).cache()
    // FIXME: this `count` op reduces the time this method executes from
    // > 7 minutes to ~30 s
    //training.count
    training
  }
  def testSet(sc : SparkContext) : RDD[LabeledPoint] = splits(sc)(1).cache()


  //def train(modelParams: P) : M

  def evaluateModel(model : M,
                    data :RDD[LabeledPoint]) : MulticlassMetrics

  def persistModel(sc : SparkContext, model: M)
  def loadPersistedModel(sc : SparkContext) : M

  def predict(features: Vector, model: M) : Double

  // NOTE: use this to test out a bunch of different training parameters
  // and return a list of the parameters with performance metrics
  //def exploreTraining(trainingData:RDD[LabeledPoint],
                      //testData: RDD[LabeledPoint]) : Seq[Perf[P]]

  //def exploreTrainingResults : Seq[Perf[P]]
}
