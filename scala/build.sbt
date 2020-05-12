name := "arctern_scala"

version := "0.1"

scalaVersion := "2.12.11"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.8" % "test"
libraryDependencies += "org.locationtech.jts" % "jts-core" % "1.16.1"
libraryDependencies += "org.apache.spark" %% "spark-core" % "3.0.0-preview2"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.0.0-preview2"
