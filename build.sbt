lazy val root = (project in file(".")).
settings(
  name := "ML Playground",
  version := "0.0.1",
  scalaVersion := "2.11.7",

  mainClass := Some("ExploreKDD"),

   libraryDependencies += "org.apache.spark" %% "spark-core" % "1.5.2" ,
   libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.5.2",
   libraryDependencies += "com.databricks" %% "spark-csv" % "1.3.0"
)
