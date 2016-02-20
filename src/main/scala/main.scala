package spark_ml_playground

import spark_ml_playground.AcuteInflammation

import org.apache.log4j.Logger
import org.apache.log4j.Level


object Main {

  def main(args: Array[String]) = {
    Logger.getRootLogger().setLevel(Level.WARN)
    AcuteInflammation.trainTestAndEval
  }
}
