package spark_ml_playground

import java.util.Calendar

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


  def allData(sc : SparkContext) = KDD.cachedModelData(sc)

  def train(modelParams : SGDModelParams,
            trainingData: RDD[LabeledPoint]) : LogisticRegressionModel = {

    var model = new LogisticRegressionWithSGD()
      model.optimizer.
      setNumIterations(modelParams.numIterations).
      setRegParam(modelParams.regParam)
    model.run(trainingData)
  }

  private def generateModelParams = {
    for(regParam <- (0.00001 to 0.01 by 0.0005);
      numIterations <- (10 to 1000 by 100) ) yield SGDModelParams(regParam, numIterations)
  }

  //def exploreTraining(trainingData: RDD[LabeledPoint],
                      //testData: RDD[LabeledPoint],
                      //modelParams: RDD[SGDModelParams]) = {

    //modelParams.map { modelParam =>

      //println("about to train")
      //var model = new LogisticRegressionWithSGD()
        //model.optimizer.
        //setNumIterations(modelParam.numIterations).
        //setRegParam(modelParam.regParam)
      //model.run(trainingSet)
      ////val model = train(modelParam)
      ////val metrics = evaluateModel(model, testData)

      ////FIXME: this is for debugging
      ////println(modelParam.regParam + "," + modelParam.numIterations + "," + metrics.weightedRecall + "," + metrics.weightedPrecision)
      ////Perf[SGDModelParams](modelParam, metrics.weightedRecall, metrics.weightedPrecision)
      //Perf[SGDModelParams](modelParam, 0.1, 0.2)
    //}
  //}

  def everythingInOne(sc : SparkContext) = {
    // NOTE: this needs to be brought into a local variable or else the
    // training step completely hangs
    val trs_ = trainingSet(sc)
    val tes_ = testSet(sc)
    // required?!?
    trs_.count
    tes_.count

    val bcts = sc.broadcast(trs_).value.cache()
    val bcss = sc.broadcast(tes_).value.cache()

    // 1. Generate model params
    // 2. Develop models on each param set
    println("=== training "+ generateModelParams.length + " models")
    val results = generateModelParams.par.map { modelP =>
      println("=== training "+modelP+" ===")
      val model = train(modelP, bcts)
      (modelP, model)
    }.map { case(modelP, model) =>
      // 3. test model
      println("=== testing "+modelP+" ===")
      val predictionLabel = bcss.map {
          case LabeledPoint(label, features) =>
            val prediction = predict(features, model)
            (prediction, label)
      }
      (predictionLabel, modelP)
    }.map { case(modelPredictions, modelP) =>
      println("=== evaling "+modelP+" ===")
      // 4. Collect model evaluation metrics
      val metrics = new MulticlassMetrics(modelPredictions)

      Perf[SGDModelParams](modelP, metrics.weightedRecall, metrics.weightedPrecision)
    }.toList
    val rddResults = sc.parallelize(results)
    rddResults.saveAsTextFile("hdfs://spark3.thedevranch.net/model_results/results-" + Calendar.getInstance().getTimeInMillis)

    rddResults.takeOrdered(10)(Ordering[Double].reverse.on(res => res.wPrecision)).foreach { res =>
      println(res.toCSV)
    }
  }

  def persistModel(sc: SparkContext, model: LogisticRegressionModel) = {
    model.save(sc, modelPersistancePath)
  }

  def loadPersistedModel(sc : SparkContext) : LogisticRegressionModel = {
    LogisticRegressionModel.load(sc, modelPersistancePath)
  }

  def predict(features: Vector, model: LogisticRegressionModel) : Double = {
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
