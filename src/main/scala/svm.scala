package spark_ml_playground

import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.regression.LabeledPoint

case class Perf(regParam: BigDecimal, numIterations: Int, wRecall: Double, wPrecision: Double)

class PerformanceSummary() {
  var perfSummaries = List[Perf]()

  def addSummary(perf: Perf) = perfSummaries = perfSummaries :+ perf

  def sortSummaries = (perfSummaries sortBy (_.wPrecision))

  def bestModel = sortSummaries.last
  def worstModel = sortSummaries.head

  def printSummary = {
    sortSummaries.foreach{ x =>
      println(x)
    }
  }
}

object SVM {

  def train(trainingData: org.apache.spark.rdd.RDD[LabeledPoint], numIterations: Int) : SVMModel = {
    val model = buildModel(10, BigDecimal(1.0))
    model.run(trainingData)
  }

  def exploreTraining(trainingData: org.apache.spark.rdd.RDD[LabeledPoint],
                      testData: org.apache.spark.rdd.RDD[LabeledPoint]) = {

    var perfSummary =  new PerformanceSummary()

    for(regParam <- (BigDecimal(0.1) to BigDecimal(3.0) by 0.5);
        numIterations <- (10 to 20 by 10) ) {


      val model = buildModel(numIterations, regParam)
        .run(trainingData)

      val pred = SVM.predict(model, testData)
      val metrics = Main.modelPerformance(pred)

      perfSummary.addSummary(
        Perf(regParam, numIterations, metrics.weightedRecall, metrics.weightedPrecision)
      )
    }

    println("=== Worst Mdoel: " + perfSummary.worstModel)
    println("=== Best Mdoel: " + perfSummary.bestModel)
  }

  private def buildModel(numInterations: Int, regParam: BigDecimal) = {
      //setUpdater(new L1Updater)
    val model = new SVMWithSGD()
      model.optimizer.
      setNumIterations(numInterations).
      setRegParam(0.1)
    model
  }

  def predict(model : SVMModel,
              data : org.apache.spark.rdd.RDD[LabeledPoint]) = {

              // Clear the default threshold.
              // model.clearThreshold()

    data.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

  }
}
