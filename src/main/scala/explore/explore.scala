package spark_ml_playground

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

trait Explore[
  D <: DataSource,
  P <: ModelParams
] {
  lazy val sc : SparkContext = new SparkSetup().sc
  lazy val seedVal : Long = 11L
  lazy val trainingSetRatio : Double = 0.8
  lazy val testSetRatio : Double = 1.0 - trainingSetRatio

  def allData : RDD[LabeledPoint]
  def splits = allData.randomSplit(Array(trainingSetRatio, testSetRatio), seed = seedVal)
  def trainingSet : RDD[LabeledPoint] = splits(0).cache()
  def testSet : RDD[LabeledPoint] = splits(1).cache()

  def results : Seq[Perf[P]]

}
