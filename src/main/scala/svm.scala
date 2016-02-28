package spark_ml_playground

import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.regression.LabeledPoint

case class Perf(modelParams: SVMModelParams, wRecall: Double, wPrecision: Double)

case class SVMModelParams( regParam: Double,
                           numIterations: Int) {

  override def toString = "regularization value: " + regParam + " num. iterations " + numIterations
}

class PerformanceSummary(perfSummaries: Seq[Perf]) {

  def sortSummaries = (perfSummaries sortBy (_.wPrecision))

  def bestModel = modelPerfToString(sortSummaries.last)
  def worstModel = modelPerfToString(sortSummaries.head)

  def bestModelParam = sortSummaries.last.modelParams

  private def modelPerfToString(perf: Perf) =  perf.modelParams.toString + " resulted in weighted recall " + perf.wRecall + " and weighted Precision " + perf.wPrecision

  def printSummary = {
    sortSummaries.foreach{ x =>
      println(x)
    }
  }
}

object SVM {

  def train(trainingData: org.apache.spark.rdd.RDD[LabeledPoint], modelParams: SVMModelParams) : SVMModel = {
    // FIXME: apply the modelParams to the build model call
    val model = buildModel( modelParams )
    model.run(trainingData)
  }

  private def generateModelParams = {
    for(regParam <- (0.1 to 3.0 by 0.5);
      numIterations <- (10 to 20 by 10) ) yield SVMModelParams(regParam, numIterations)

  }
  def exploreTraining(trainingData: org.apache.spark.rdd.RDD[LabeledPoint],
                      testData: org.apache.spark.rdd.RDD[LabeledPoint]): PerformanceSummary = {

    // FIXME: this `count` op reduces the time this method executes from
    // > 7 minutes to ~30 s
    trainingData.count

    val modelIters = generateModelParams.map { modelParam =>

      val model = buildModel( modelParam )
        .run(trainingData)

      val pred = SVM.evaluateModel(model, testData)
      val metrics = Main.modelPerformance(pred)

      Perf(modelParam, metrics.weightedRecall, metrics.weightedPrecision)
    }
    new PerformanceSummary(modelIters)
  }

  private def buildModel(modelParams: SVMModelParams) = {
      //setUpdater(new L1Updater)
    val model = new SVMWithSGD()
      model.optimizer.
      setNumIterations(modelParams.numIterations).
      setRegParam(modelParams.regParam)
    model
  }

  def evaluateModel(model : SVMModel,
                    data : org.apache.spark.rdd.RDD[LabeledPoint]) = {

              // Clear the default threshold.
              // model.clearThreshold()

    data.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

  }
}
