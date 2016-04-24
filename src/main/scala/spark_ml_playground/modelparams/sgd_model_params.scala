package spark_ml_playground.modelparams

case class SGDModelParams( regParam: Double,
                           numIterations: Int) extends ModelParams {

  override def toString = "regularization value: " + regParam + " num. iterations " + numIterations
  def toCSV = regParam + "," + numIterations
}
