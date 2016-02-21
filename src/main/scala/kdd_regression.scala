package spark_ml_playground

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

import org.apache.log4j.Logger
import org.apache.log4j.Level

object KDD {

  def matchYesNo(vari: String) : Int = vari match {
    case "yes" => 1
    case "no"  => 0
  }

  def nephritisPoint(dataString: String) : LabeledPoint = {
      val data = dataString.split("\t")
      val patientTemp = data(0).replaceAll(",", ".").toDouble
      val patientNausea = matchYesNo( data(1) )
      val patientLumbarPain = matchYesNo( data(2) )
      val patientUrinePushing = matchYesNo( data(3) )
      val patientMicturitionPain = matchYesNo( data(4) )
      val patientBurning = matchYesNo( data(5) )
      val outcomeInflamation = matchYesNo( data(6) )
      val outcomeNephritis  = matchYesNo( data(7) )

      val vec = Vectors.dense(patientTemp,
                              patientNausea,
                              patientLumbarPain,
                              patientUrinePushing,
                              patientMicturitionPain,
                              patientBurning
                            )
      LabeledPoint(outcomeNephritis, vec)
  }

  def train(trainingData: org.apache.spark.rdd.RDD[LabeledPoint]) : LogisticRegressionModel = {
    new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
    //new LogisticRegressionWithSGD()
      .run(trainingData)
  }

  def predict(model : LogisticRegressionModel,
    data : org.apache.spark.rdd.RDD[LabeledPoint]) = {

    data.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
  }

  def modelPerformance(predictionAndLabels: org.apache.spark.rdd.RDD[(Double, Double)]) = {
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)

    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    println("[Model Perf] AUC: " + metrics.areaUnderROC)
    println("[Model Perf] Confusion Matrix: ")

    val multMetrics = new MulticlassMetrics(predictionAndLabels)
    println(multMetrics.confusionMatrix)
    metrics
  }

  def trainTestAndEval(sc : SparkContext) = {
    val sqlContext = new SQLContext(sc)
    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("inferSchema", "true")
      .load("hdfs://spark3.thedevranch.net/classifications-datasets/kddcup.data_10_percent")

      //.fit(df, ("C1" -> "transportProtoIndex"), ("C2" -> "protoIndex"))
    val mappers: List[(String, String)] = List(
      "C1" -> "transportProtoIndex",
      "C2" -> "protoIndex",
      "C3" -> "idunnoIndex",
      "C41" -> "outcome"
     )

   var mutDataFrame : DataFrame = df
   val transformed = mappers.map{ m =>
     val strIdx = new StringIndexer()
     .setInputCol(m._1)
     .setOutputCol(m._2)
     mutDataFrame = transformIt(strIdx, mutDataFrame)
   }
   //mutDataFrame.printSchema()
   // casting to doubles is required for stuff not to blow up
   val modelData = mutDataFrame.selectExpr(
     "cast(CASE C41 WHEN 'normal.' THEN 0.0 ELSE 1.0 END as DOUBLE) as outcomeBin",
     "cast(C4 as DOUBLE) as C4",
     "transportProtoIndex",
     "protoIndex",
     "idunnoIndex"
   )

   modelData.show()

   //NOTE: if the select statement from creating this df changes, update `rowToLabeledPoint`
   val splits = modelData
     .map{row =>
       //println( row.schema )
       val vec = Vectors.dense(
         row.getDouble(1),
         row.getDouble(2),
         row.getDouble(3),
         row.getDouble(4)
       )
       LabeledPoint(row.getDouble(0), vec)
     }
     .randomSplit(Array(0.8, 0.2), seed = 11L)
   val training = splits(0).cache()
   val test = splits(1)

    println("===== TRAINING =====")
    println("data set count: " + training.count)
    val model = train(training)

    println("===== PREDICT =====")
    println("weights: " + model.weights)
    val predictionAndLabels = predict(model, test)

    println("===== MODEL PERF =====")
    modelPerformance(predictionAndLabels)
  }

  def transformRowToLabeledPoint(row: Row) = LabeledPoint(row.getDouble(0),
       Vectors.dense(
         row.getDouble(1),
         row.getDouble(2),
         row.getDouble(3),
         row.getDouble(4)
      )
  )

  def transformIt(strIdxer: StringIndexer, df: DataFrame) : DataFrame = {
    strIdxer.fit(df).transform(df)
  }

  def trainTestAndEvalOLD(sc : SparkContext) = {

    // NOTE: dataset is @ https://archive.ics.uci.edu/ml/datasets/URL+Reputation
    //val data = MLUtils.loadLibSVMFile(sc, "hdfs://spark3.thedevranch.net/classifications-datasets/diagnosis.data")

    val diagnosisData = sc.textFile("hdfs://spark3.thedevranch.net/classifications-datasets/kddcup.data_10_percent")
    // https://archive.ics.uci.edu/ml/machine-learning-databases/acute/
    val nephritis = diagnosisData.map(row => nephritisPoint(row) )
    val splits = nephritis.randomSplit(Array(0.8, 0.2))
    val training = splits(0)//.cache()
    val test = splits(1) //.cache()

    println("===== TRAINING =====")
    println("data set count: " + training.count)
    val model = train(training)


    // this syntax is not great. It indicates that index 1, has 2 categories
    // FIXME - make sure categorical vars are used
    val categoricalFeaturesInfo = Map[Int, Int]((1 -> 2), (2 -> 2))


    println("===== PREDICT =====")
    println("weights: " + model.weights)
    val predictionAndLabels = predict(model, test)

    // output some cases to see how well we did
    //predictionAndLabels.take(5).foreach(println)

    println("===== MODEL PERF =====")
    modelPerformance(predictionAndLabels)
  }
}
