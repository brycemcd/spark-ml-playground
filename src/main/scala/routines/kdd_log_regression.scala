package spark_ml_playground

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._

import org.apache.log4j.Logger
import org.apache.log4j.Level


//object KddLogRegressionRoutine {
  //def main(args: Array[String]) = {
    //val conf = new SparkConf().setAppName("KDDLogRegressionRoutine")
    //val sc = new SparkContext(conf)
    //val allModelParams = KddLogisticRegression.exploreTraining(sc,
      //KddLogisticRegression.minGenerateModelParams)
  //}
//}

//object KddSVMRoutine{
  //def main(args: Array[String]) = {
    //val conf = new SparkConf().setAppName("KDDLogRegressionRoutine")
    //val sc = new SparkContext(conf)
    //val allModelParams = KddSVM.exploreTraining(sc,
      //KddSVM.minGenerateModelParams)
  //}
//}

object KddAllModelsRoutine{
  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName("KDDAllModelsRoutine")
    val sc = new SparkContext(conf)

    val minParams = KddLogisticRegression.maxGenerateModelParams
    // training step completely hangs
    val trs_ = KddLogisticRegression.trainingSet(sc)
    val tes_ = KddLogisticRegression.testSet(sc)
    // required?!?
    trs_.count
    tes_.count

    val bcts = sc.broadcast(trs_).value.cache()
    val bcss = sc.broadcast(tes_).value.cache()

    bcts.count
    bcss.count

    val logResults = KddLogisticRegression.exploreTraining(sc, minParams, bcts, bcss)
    KddLogisticRegression.persistExploratoryResults(logResults)

    val svmResults = KddSVM.exploreTraining(sc, minParams, bcts, bcss)
    KddSVM.persistExploratoryResults(svmResults)
  }
}
