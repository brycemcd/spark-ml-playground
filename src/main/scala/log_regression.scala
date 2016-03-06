package spark_ml_playground

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD}

import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Vector

case class LRModelParams(numClasses: Int) extends ModelParams

object LogisticRegression extends Model[
  LogisticRegressionModel,
  SVMModelParams
] {

  def train(trainingData: org.apache.spark.rdd.RDD[LabeledPoint],
            modelParams : SVMModelParams) : LogisticRegressionModel = {
    //new LogisticRegressionWithSGD()
    buildModel(modelParams)
      .run(trainingData)
  }

  private def buildModel(modelParams: SVMModelParams) = {
    //val model = new LogisticRegressionWithLBFGS()
    //model.setNumClasses(modelParams.numClasses)
    var model = new LogisticRegressionWithSGD()
      model.optimizer.
      setNumIterations(modelParams.numIterations).
      setRegParam(modelParams.regParam)
    model
  }

  private def generateModelParamsOLD : Seq[LRModelParams] = {
    for(numClasses <- (2 to 2)) yield LRModelParams(numClasses)
  }

  private def generateModelParams : Seq[SVMModelParams] = {
    //for(regParam <- (0.00001 to 1.00 by 0.0005);
      //numIterations <- (100 to 3000 by 300) ) yield SVMModelParams(regParam, numIterations)
    for(regParam <- (0.00001 to 0.0001 by 0.0005);
      numIterations <- (100 to 200 by 100) ) yield SVMModelParams(regParam, numIterations)
  }

  def exploreTraining(trainingData: org.apache.spark.rdd.RDD[LabeledPoint],
                      testData: org.apache.spark.rdd.RDD[LabeledPoint]) : Seq[Perf[SVMModelParams]] = {

    // FIXME: this `count` op reduces the time this method executes from
    // > 7 minutes to ~30 s
    trainingData.count

    generateModelParams.map { modelParam =>

      val model = buildModel( modelParam )
        .run(trainingData)

      val metrics = this.evaluateModel(model, testData)

      println(modelParam.regParam + "," + modelParam.numIterations + "," + metrics.weightedRecall + "," + metrics.weightedPrecision)
      Perf[SVMModelParams](modelParam, metrics.weightedRecall, metrics.weightedPrecision)
    }
  }

  def persistModel(sc: SparkContext,
                  name: String,
                  modelParam: SVMModelParams,
                  trainingData: org.apache.spark.rdd.RDD[LabeledPoint]) = {

      trainingData.count
      val model = buildModel( modelParam )
        .run(trainingData)

      model.save(sc, s"hdfs://spark3.thedevranch.net/ml-models/$name")
  }

  def predict(sc: SparkContext,
              name: String,
              features: Vector) = {

    val model = LogisticRegressionModel.load(sc, s"hdfs://spark3.thedevranch.net/ml-models/$name")
    model.predict(features)
  }

  def evaluateModel(model : LogisticRegressionModel,
                    data : org.apache.spark.rdd.RDD[LabeledPoint]) = {

    val predictionAndLabels = data.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    new MulticlassMetrics(predictionAndLabels)
  }
}
