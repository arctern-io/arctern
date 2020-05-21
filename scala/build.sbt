name := "arctern_scala"

version := "0.1"

scalaVersion := "2.12.11"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.8" % "test"
libraryDependencies += "org.locationtech.jts" % "jts-core" % "1.16.1"

val sparkVersion = Option(System.getProperty("sparkVersion")).getOrElse("3.0.0-preview2")
if (sparkVersion == "3.0.0-preview2") {
  println("Build arctern with spark-3.0.0-preview2")
  libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
  libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion
} else if (sparkVersion == "2.4.5") {
  println("Build arctern with spark-2.4.5")
  libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
  libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion
} else {
  println("Unrecognized spark version, build arctern with default version: spark-3.0.0-preview2")
  libraryDependencies += "org.apache.spark" %% "spark-core" % "spark-3.0.0-preview2"
  libraryDependencies += "org.apache.spark" %% "spark-sql" % "spark-3.0.0-preview2"
}
