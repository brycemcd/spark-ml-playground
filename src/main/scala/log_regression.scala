package spark_ml_playground

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel, LogisticRegressionWithSGD}

import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint

object LogisticRegression {

  def train(trainingData: org.apache.spark.rdd.RDD[LabeledPoint], numClasses : Int) : LogisticRegressionModel = {
    new LogisticRegressionWithLBFGS()
      .setNumClasses(numClasses)
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
}
