//TODO: learn how to namespace
package spark_ml_playground

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import com.databricks.spark.csv
import org.apache.spark.rdd.RDD

trait DataSet {
  val uniqueDataCacheName: String

  //def cachedModelData(sc : SparkContext) : RDD[LabeledPoint]
  //def prepareRawData(sc: SparkContext) : RDD[LabeledPoint]
}

abstract class KDD extends DataSet {}

object KDD extends DataSet {
  val uniqueDataCacheName: String = "kdd"
  private val dataSetStoragePath = "hdfs://spark3.thedevranch.net/ml-data/"+uniqueDataCacheName +".modelData"

  def prepareRawData(sc : SparkContext) = {
    val sqlContext = new SQLContext(sc)
    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("inferSchema", "true")
      .load("hdfs://spark3.thedevranch.net/classifications-datasets/kddcup.data_10_percent")
      //.load("hdfs://spark3.thedevranch.net/classifications-datasets/kddcup.data")

    val mappers: List[(String, String)] = List(
      ("C1" , "transportProtoIndex"),
      ("C2" , "protoIndex"),
      ("C3" , "idunnoIndex"),
      ("C41" , "outcome")
     )

   var mutDataFrame : DataFrame = df
   // TODO: this is inefficient
   val transformed = mappers.map{ m =>
     val strIdx = new StringIndexer()
     .setInputCol(m._1)
     .setOutputCol(m._2)
     mutDataFrame = transformIt(strIdx, mutDataFrame)
   }
   //mutDataFrame.show()

   val modelData = mutDataFrame.selectExpr(
     "cast(CASE C41 WHEN 'normal.' THEN 0.0 ELSE 1.0 END as DOUBLE) as outcomeBin",
     "cast(C4 as DOUBLE) as C4",
     "cast(C5 as DOUBLE) as C5",
     "cast(C6 as DOUBLE) as C6",
     "cast(C7 as DOUBLE) as C7",
     "cast(C8 as DOUBLE) as C8",
     "cast(C9 as DOUBLE) as C9",
     "cast(C10 as DOUBLE) as C10",
     "cast(C11 as DOUBLE) as C11",
     "cast(C12 as DOUBLE) as C12",
     "cast(C13 as DOUBLE) as C13",
     "cast(C14 as DOUBLE) as C14",
     "cast(C15 as DOUBLE) as C15",
     "cast(C16 as DOUBLE) as C16",
     "cast(C17 as DOUBLE) as C17",
     "cast(C18 as DOUBLE) as C18",
     "cast(C19 as DOUBLE) as C19",
     "cast(C20 as DOUBLE) as C20",
     "cast(C21 as DOUBLE) as C21",
     "cast(C22 as DOUBLE) as C22",
     "cast(C23 as DOUBLE) as C23",
     "cast(C24 as DOUBLE) as C24",
     "cast(C25 as DOUBLE) as C25",
     "cast(C26 as DOUBLE) as C26",
     "cast(C27 as DOUBLE) as C27",
     "cast(C28 as DOUBLE) as C28",
     "cast(C29 as DOUBLE) as C29",
     "cast(C30 as DOUBLE) as C30",
     "cast(C31 as DOUBLE) as C31",
     "cast(C32 as DOUBLE) as C32",
     "cast(C33 as DOUBLE) as C33",
     "cast(C34 as DOUBLE) as C34",
     "cast(C35 as DOUBLE) as C35",
     "cast(C36 as DOUBLE) as C36",
     "cast(C37 as DOUBLE) as C37",
     "cast(C38 as DOUBLE) as C38",
     "cast(C39 as DOUBLE) as C39",
     "cast(C40 as DOUBLE) as C40",
     "transportProtoIndex",
     "protoIndex",
     "idunnoIndex"
   )

   //modelData.show()

   //NOTE: if the select statement from creating this df changes, update `rowToLabeledPoint`
    modelData.write.parquet(dataSetStoragePath)
    modelData
     .map(row => transformRowToLabeledPoint(row) )
  }

  def cachedModelData(sc : SparkContext) = {
    val sqlContext = new SQLContext(sc)

    val modelData = sqlContext.read.parquet(dataSetStoragePath)
    modelData
     .map(row => transformRowToLabeledPoint(row) )
  }

  private def transformRowToLabeledPoint(row: Row) = LabeledPoint(row.getDouble(0),
    Vectors.dense(
      row.getDouble(1),
      row.getDouble(2),
      row.getDouble(3),
      row.getDouble(4),
      row.getDouble(5),
      row.getDouble(6),
      row.getDouble(7),
      row.getDouble(8),
      row.getDouble(9),
      row.getDouble(10),
      row.getDouble(11),
      row.getDouble(12),
      row.getDouble(13),
      row.getDouble(14),
      row.getDouble(15),
      row.getDouble(16),
      row.getDouble(17),
      row.getDouble(18),
      row.getDouble(19),
      row.getDouble(20),
      row.getDouble(21),
      row.getDouble(22),
      row.getDouble(23),
      row.getDouble(24),
      row.getDouble(25),
      row.getDouble(26),
      row.getDouble(27),
      row.getDouble(28),
      row.getDouble(29),
      row.getDouble(30),
      row.getDouble(31),
      row.getDouble(32),
      row.getDouble(33),
      row.getDouble(34),
      row.getDouble(35),
      row.getDouble(36),
      row.getDouble(37),
      row.getDouble(38),
      row.getDouble(39),
      row.getDouble(40)
    )
  )

  def transformIt(strIdxer: StringIndexer, df: DataFrame) : DataFrame = {
    strIdxer.fit(df).transform(df)
  }
}
