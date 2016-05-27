package spark_ml_playground.modelparams

import org.apache.spark.mllib.evaluation.MulticlassMetrics

trait ModelParams {
  def toCSV : String
}

trait PerfPrinter {
}

// TODO: Really need to get this P out of here. It affects Model + Explore
case class Perf[P <: ModelParams](modelParams : P, wRecall: Double,
                                  wPrecision: Double,
                                  recall : Double = 0.0) {
  override def toString = modelParams.toString + " " + " wRecall: " + wRecall + " wPrecision: " + wPrecision + " recall: " + recall

  def toCSV = modelParams.toCSV + "," + wRecall + "," + wPrecision + "\n"
}
