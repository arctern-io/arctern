/*
 * Copyright (C) 2019-2020 Zilliz. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.spark.sql.Row
import org.apache.spark.sql.arctern.GeometryUDT
import org.apache.spark.sql.arctern.expressions.IndexedJoin
import org.apache.spark.sql.arctern.index.{IndexBuilder, RTreeIndex}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.locationtech.jts.io.WKTReader

class IndexTest extends AdapterTest {

  test("test index") {
    val index = new RTreeIndex
    val point1 = new WKTReader().read("POINT(-100 -40)")
    val point2 = new WKTReader().read("POINT(100 -40)")
    val point3 = new WKTReader().read("POINT(-100 40)")
    val point4 = new WKTReader().read("POINT(100 40)")
    val point5 = new WKTReader().read("POINT(100 0)")

    val polygon1 = new WKTReader().read("POLYGON((-180 -90, 0 -90, 0 0, -180 0, -180 -90))")
    val polygon2 = new WKTReader().read("POLYGON((0 -90, 180 -90, 180 0, 0 0, 0 -90))")
    val polygon3 = new WKTReader().read("POLYGON((-180 0, 0 0, 0 90, -180 90, -180 0))")
    val polygon4 = new WKTReader().read("POLYGON((0 0, 180 0, 180 90, 0 90, 0 0))")

    index.insert(polygon1.getEnvelopeInternal, polygon1)
    index.insert(polygon2.getEnvelopeInternal, polygon2)
    index.insert(polygon3.getEnvelopeInternal, polygon3)
    index.insert(polygon4.getEnvelopeInternal, polygon4)

    var env = point1.getEnvelopeInternal
    var result = index.query(env).toString
    println(result)
    env = point2.getEnvelopeInternal
    result = index.query(env).toString
    println(result)
    env = point3.getEnvelopeInternal
    result = index.query(env).toString
    println(result)
    env = point4.getEnvelopeInternal
    result = index.query(env).toString
    println(result)
    env = point5.getEnvelopeInternal
    result = index.query(env).toString
    println(result)
  }

  test("Indextest") {
    val index = new IndexBuilder("RTREE")
    val data = Seq(
      Row(1, "POINT(-100 -40)", "POLYGON((-180 -90, -20 -60, 0 0, -180 0, -180 -90))"),
      Row(2, "POINT(100 -40)", "POLYGON((0 -90, 180 -90, 180 0, 0 0, 0 -90))"),
      Row(3, "POINT(-100 40)", "POLYGON((-180 0, 0 0, 0 90, -180 90, -180 0))"),
      Row(4, "POINT(100 40)", "POLYGON((0 0, 180 0, 180 90, 0 90, 0 0))")
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("idx", IntegerType, nullable = false), StructField("geo1", StringType, nullable = false), StructField("geo2", StringType, nullable = false)))
    val df = spark.createDataFrame(rdd_d, schema)
    val geo1 = df.select("geo1").collect.map(row => GeometryUDT.FromWkt(row.getString(0)))
    val geo2 = df.select("geo2").collect.map(row => GeometryUDT.FromWkt(row.getString(0)))
    index.insert(geo2)

    val broadcastVar = spark.sparkContext.broadcast(index)
    val joincase = new IndexedJoin(broadcastVar)
    val result = joincase.join(geo1)
    result.foreach { out =>
      println(out.toString)
    }
  }

  test("test1 big data") {
    val resourceFolder = System.getProperty("user.dir") + "/src/test/resources/"
    val PointLocation = resourceFolder + "point.csv"
    val PolygonLocation = resourceFolder + "polygon.csv"

    val conf_array = spark.sparkContext.getConf.getAll
    for (x <- conf_array) {
      println(x)
    }

    val df0 = spark.read.format("csv")
      .option("sep", ";")
      .option("header", "false")
      .load(PointLocation)
    val df1 = spark.read.format("csv")
      .option("sep", ";")
      .option("header", "false")
      .load(PolygonLocation)

    val geo1 = df0.select("_c0").collect.map(row => GeometryUDT.FromWkt(row.getString(0)))
    val geo2 = df1.select("_c0").collect.map(row => GeometryUDT.FromWkt(row.getString(0)))
    println("preference test 1++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    val t1 = System.currentTimeMillis
    val index = new IndexBuilder("RTREE")
    index.insert(geo2)
    val t2 = System.currentTimeMillis
    println((t2 - t1) / 1000.0 + " secs")
    println("preference test 2++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    val broadcastVar = spark.sparkContext.broadcast(index)
    println(broadcastVar.value)
    val joincase = new IndexedJoin(broadcastVar)
    val t3 = System.currentTimeMillis
    println((t3 - t2) / 1000.0 + " secs")
    println("preference test 3++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    val result = joincase.join(geo1)
    val t4 = System.currentTimeMillis
    println((t4 - t3) / 1000.0 + " secs")
    println("preference test 4++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    //    result.foreach{ out =>
    //      println(out.toString)
    //    }
  }

}
