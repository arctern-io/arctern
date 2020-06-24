name := "arctern_scala"

version := "0.1"

scalaVersion := "2.12.11"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.8" % "test"
libraryDependencies += "org.locationtech.jts" % "jts-core" % "1.16.1"
libraryDependencies += "org.wololo" % "jts2geojson" % "0.12.0"

val sparkVersion = Option(System.getProperty("sparkVersion")).getOrElse("3.0.0")
if (sparkVersion == "3.0.0") {
  println("Build arctern with spark-3.0.0")
  libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
  libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion
} else if (sparkVersion == "2.4.5") {
  println("Build arctern with spark-2.4.5")
  libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
  libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion
} else {
  println("Unrecognized spark version, build arctern with default version: spark-3.0.0")
  libraryDependencies += "org.apache.spark" %% "spark-core" % "3.0.0"
  libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.0.0"
}

resolvers += "Open Source Geospatial Foundation Repository" at "https://repo.osgeo.org/repository/release/"
libraryDependencies ++= Seq(
  "org.geotools" % "gt-main" % "23.1",
  "org.geotools" % "gt-referencing" % "23.1",
  "org.geotools" % "gt-epsg-hsql" % "23.1"
)

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

