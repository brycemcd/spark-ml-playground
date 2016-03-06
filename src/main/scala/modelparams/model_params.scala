package spark_ml_playground

abstract class ModelParams

// TODO: Really need to get this P out of here. It affects Model + Explore
case class Perf[P <: ModelParams](modelParams : P, wRecall: Double, wPrecision: Double)
