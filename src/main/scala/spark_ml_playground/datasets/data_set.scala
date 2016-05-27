package spark_ml_playground.datasets

trait DataSet {
  val uniqueDataCacheName: String

  //def cachedModelData(sc : SparkContext) : RDD[LabeledPoint]
  //def prepareRawData(sc: SparkContext) : RDD[LabeledPoint]
}
