//TODO learn how to namespace
package spark_ml_playground

import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.regression.LabeledPoint

case class SVMModelParams( regParam: Double,
                           numIterations: Int) extends ModelParams {

  override def toString = "regularization value: " + regParam + " num. iterations " + numIterations
}



object SVM extends Model[
  SVMModel,
  SVMModelParams
] {

  def train(trainingData: org.apache.spark.rdd.RDD[LabeledPoint], modelParams: SVMModelParams) : SVMModel = {
     buildModel(modelParams)
      .run(trainingData)
  }

  private def generateModelParams : Seq[SVMModelParams] = {
    for(regParam <- (0.1 to 3.0 by 0.5);
      numIterations <- (10 to 20 by 10) ) yield SVMModelParams(regParam, numIterations)

  }

  def exploreTraining(trainingData: org.apache.spark.rdd.RDD[LabeledPoint],
                      testData: org.apache.spark.rdd.RDD[LabeledPoint]) = {

    // FIXME: this `count` op reduces the time this method executes from
    // > 7 minutes to ~30 s
    trainingData.count

    val modelIters = generateModelParams.map { modelParam =>

      val model = buildModel( modelParam )
        .run(trainingData)

      val metrics = SVM.evaluateModel(model, testData)

      Perf(modelParam, metrics.weightedRecall, metrics.weightedPrecision)
    }
    //new PerformanceSummary(modelIters)
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
                    data : org.apache.spark.rdd.RDD[LabeledPoint]) : MulticlassMetrics = {

    val predictionAndLabels = data.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    new MulticlassMetrics(predictionAndLabels)
  }
}
