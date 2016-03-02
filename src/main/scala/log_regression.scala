package spark_ml_playground

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD}

import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics

case class LRModelParams(numClasses: Int) extends ModelParams

object LogisticRegression extends Model[
  LogisticRegressionModel,
  LRModelParams
] {

  def train(trainingData: org.apache.spark.rdd.RDD[LabeledPoint],
            modelParams : LRModelParams) : LogisticRegressionModel = {
    //new LogisticRegressionWithSGD()
    buildModel(modelParams)
      .run(trainingData)
  }

  private def buildModel(modelParams: LRModelParams) = {

    //new LogisticRegressionWithSGD()
    new LogisticRegressionWithLBFGS()
      .setNumClasses(modelParams.numClasses)
  }

  private def generateModelParams : Seq[LRModelParams] = {
    for(numClasses <- (0 to 2)) yield LRModelParams(numClasses)
  }

  def exploreTraining(trainingData: org.apache.spark.rdd.RDD[LabeledPoint],
                      testData: org.apache.spark.rdd.RDD[LabeledPoint]) = {

    // FIXME: this `count` op reduces the time this method executes from
    // > 7 minutes to ~30 s
    trainingData.count

    val modelIters = generateModelParams.map { modelParam =>

      val model = buildModel( modelParam )
        .run(trainingData)

      val metrics = this.evaluateModel(model, testData)

      Perf[LRModelParams](modelParam, metrics.weightedRecall, metrics.weightedPrecision)
    }
    //new PerformanceSummary(modelIters)
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
