package spark_ml_playground
case class SGDModelParams( regParam: Double,
                           numIterations: Int) extends ModelParams {

  override def toString = "regularization value: " + regParam + " num. iterations " + numIterations
  def toCSV = regParam + "," + numIterations
}
