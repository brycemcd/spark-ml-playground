package spark_ml_playground

trait ModelParams {
  def toCSV : String
}

trait PerfPrinter {
}

// TODO: Really need to get this P out of here. It affects Model + Explore
case class Perf[P <: ModelParams](modelParams : P, wRecall: Double, wPrecision: Double) {
  override def toString = modelParams.toString + " " + " wRecall: " + wRecall + " wPrecision: " + wPrecision

  def toCSV = modelParams.toCSV + "," + wRecall + "," + wPrecision + "\n"
}
