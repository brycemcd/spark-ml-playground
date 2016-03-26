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


  def allData = KDD.cachedModelData(sc)

  def train(modelParams : SGDModelParams, trainingData: RDD[LabeledPoint]) : LogisticRegressionModel = {
    var model = new LogisticRegressionWithSGD()
      model.optimizer.
      setNumIterations(modelParams.numIterations).
      setRegParam(modelParams.regParam)
    model.run(trainingData)
  }

  private def generateModelParams = {
    //for(regParam <- (0.00001 to 1.00 by 0.0005);
      //numIterations <- (100 to 3000 by 300) ) yield SGDModelParams(regParam, numIterations)
    for(regParam <- (0.00001 to 0.0001 by 0.0005);
      numIterations <- (10 to 20 by 5) ) yield SGDModelParams(regParam, numIterations)
  }

  def exploreTraining(trainingData: RDD[LabeledPoint],
                      testData: RDD[LabeledPoint],
                      modelParams: RDD[SGDModelParams]) = {

    modelParams.map { modelParam =>

      println("about to train")
      var model = new LogisticRegressionWithSGD()
        model.optimizer.
        setNumIterations(modelParam.numIterations).
        setRegParam(modelParam.regParam)
      model.run(trainingSet)
      //val model = train(modelParam)
      //val metrics = evaluateModel(model, testData)

      //FIXME: this is for debugging
      //println(modelParam.regParam + "," + modelParam.numIterations + "," + metrics.weightedRecall + "," + metrics.weightedPrecision)
      //Perf[SGDModelParams](modelParam, metrics.weightedRecall, metrics.weightedPrecision)
      Perf[SGDModelParams](modelParam, 0.1, 0.2)
    }
  }

  def everythingInOne = {
    // NOTE: this needs to be brought into a local variable or else the
    // training step completely hangs
    val trs_ = trainingSet
    val tes_ = testSet
    // required?!?
    trs_.count
    tes_.count

    val bcts = sc.broadcast(trs_).value.cache()
    val bcss = sc.broadcast(tes_).value.cache()

    // 1. Generate model params
    // 2. Develop models on each param set
    generateModelParams.par.map { modelP =>
      val model = train(modelP, bcts)
      (modelP, model)
    }.map { case(modelP, model) =>
      // 3. test model
      val predictionLabel = bcss.map {
          case LabeledPoint(label, features) =>
            val prediction = predict(features, model)
            (prediction, label)
      }
      (predictionLabel, modelP)
    }.map { case(modelPredictions, modelP) =>
      // 4. Collect model evaluation metrics
      val metrics = new MulticlassMetrics(modelPredictions)

      Perf[SGDModelParams](modelP, metrics.weightedRecall, metrics.weightedPrecision)
    }.foreach(println)
  }

  def exploreTrainingResults = {
    val modelP = generateModelParams
    val ts_ = sc.broadcast( trainingSet )
    //ts_.cache()
    println("ts_ " + ts_.value.count)
    val tes = testSet.cache()

    // 1. get all model params
    // 2. return RDD of models
    val conModels = modelP.map { modelParam =>
      println("building model")
      LogisticRegressionWithSGD.train(ts_.value, 20)
      //val model = new LogisticRegressionWithSGD()
        //model.optimizer.
        //setNumIterations(modelParam.numIterations).
        //setRegParam(modelParam.regParam)
      //model.run(ts_)
    }
    // 3. use test set for evaluation
    val testResults = conModels.map { model =>
      println("testing model")
      val predictionAndLabels = testSet.map { case LabeledPoint(label, features) =>
        val prediction = predict(features, model)
        (prediction, label)
      }
      val metrics = new MulticlassMetrics(predictionAndLabels)
      (model, metrics.weightedRecall, metrics.weightedPrecision)
    }


    conModels.foreach(println)
    //val logSummary = exploreTraining(ts, tes, modelP)
    //logSummary.collect
    //val summary = logSummary sortBy (_.wPrecision)
    //var s : String = ""
    //val f = logSummary.map{ res =>
      //s += res.toString + "\n"
      //println(s)
    //}
    //logSummary.saveAsTextFile("hdfs://spark3.thedevranch.net/model_results/results-" + Calendar.getInstance().getTimeInMillis)
    //sc.parallelize(logSummary).saveAsTextFile("hdfs://spark3.thedevranch.net/model_results/results.txt")

    //println("=== Worst Model: " + summary.head)
    //println("=== Best Model: "  + summary.last)
    //logSummary
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
