package spark_ml_playground.examples.prosper

import spark_ml_playground._
import spark_ml_playground.datasets._
import spark_ml_playground.datamodels._

import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.regression.LabeledPoint
import com.databricks.spark.csv
import org.apache.spark.rdd.RDD

import org.apache.spark.ml.feature.VectorAssembler

object Prosper extends DataSet {
  val uniqueDataCacheName = "prosper"
  private val dataSetStoragePath = "hdfs://spark3.thedevranch.net/ml-data/"+uniqueDataCacheName +".modelData"

  def main(args: Array[String]) = {
    lazy val sc : SparkContext = new SparkSetup().sc
    prepareRawData(sc)
    sc.stop()
  }

  def prepareRawData(sc: SparkContext) = {
    val sqlContext = new SQLContext(sc)

    def loancountByRatingAndStatus = {
      sqlContext.sql(
        """
        SELECT
          COUNT(*) as cnt
          , prosper_rating
          , loan_status_description
        FROM prosper_loans
        GROUP BY prosper_rating, loan_status_description
        ORDER BY cnt DESC
        """).show()
    }
    def totalLoansByStatus = {
      sqlContext.sql(
        """
        SELECT
          COUNT(*) as cnt
          , loan_status_description
        FROM prosper_loans
        GROUP BY loan_status_description
        ORDER BY cnt DESC
        """).show()
    }

    def totalLoans = sqlContext.sql("SELECT COUNT(*) as cnt FROM prosper_loans").show()

    def allDataWithBinOutcome = sqlContext.sql(
      """
      SELECT
      CASE loan_status_description WHEN 'COMPLETED' THEN 1.0 ELSE 0.0 END
      , *
      FROM prosper_loans
      WHERE loan_status_description IN ('CHARGEOFF', 'DEFAULTED', 'COMPLETED')
      """).show()

    def modelDataWithBinOutcome = sqlContext.sql(
      """
      SELECT
      CASE loan_status_description WHEN 'COMPLETED' THEN 1.0 ELSE 0.0 END AS success
      , amount_borrowed
      , borrower_rate
      , prosper_rating
      , term
      , concat(year(origination_date), 'Q', quarter(origination_date)) as cohort
      , principal_paid + principal_balance as loan_amt
      FROM prosper_loans
      WHERE loan_status_description IN ('CHARGEOFF', 'DEFAULTED', 'COMPLETED')
      """)



    val df = sqlContext.read
      .format("com.databricks.spark.csv")
      .option("inferSchema", "true")
      .option("header", "true")
      .load("hdfs://spark3.thedevranch.net/prosper/data/raw/loans/*.csv")
    df.registerTempTable("prosper_loans")

    val modelData = modelDataWithBinOutcome

    val indexer = new StringIndexer()
      .setInputCol("prosper_rating")
      .setOutputCol("prosper_rating_ind")
      .fit(modelData)

    val indexed = indexer.transform(modelData)
    val encoder = new OneHotEncoder()
      .setInputCol("prosper_rating_ind")
      .setOutputCol("prosper_rating_enc")
    val encodedData =  encoder.transform(indexed)

    val assembler = new VectorAssembler()
      .setInputCols(Array("amount_borrowed", "prosper_rating_enc"))
      .setOutputCol("features")
    val assembeled = assembler.transform(encodedData)

    val finalModel = assembeled.select("success", "features")
    finalModel.write.parquet(dataSetStoragePath)

    finalModel.map { row =>
      LabeledPoint(row.getDecimal(0).doubleValue(), row.getAs[SparseVector]("features"))
    }
  }

  def cachedModelData(sc : SparkContext) = {
    val sqlContext = new SQLContext(sc)

    val modelData = sqlContext.read.parquet(dataSetStoragePath)
    modelData.map(row =>
      LabeledPoint(row.getDecimal(0).doubleValue(), row.getAs[SparseVector]("features"))
    )
  }
}
