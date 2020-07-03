package org.apache.spark.sql.arctern

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{Column, Row, SparkSession}
import org.apache.spark.sql.arctern.expressions.{IndexedJoin, ST_Affine, ST_UnaryOp}
import org.apache.spark.sql.arctern.functions.st_within
import org.apache.spark.sql.arctern.index.RTreeIndex
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.expressions.codegen.{CodegenContext, CodegenFallback, ExprCode}
import org.apache.spark.sql.catalyst.plans.logical.BROADCAST
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.types.{AbstractDataType, BooleanType, DataType, IntegerType, StringType, StructField, StructType}
import org.locationtech.jts.geom.Geometry
import org.apache.spark.sql.functions._
import org.locationtech.jts.io.WKTReader

case class ComputeNearestRoad(inputExpressions: Seq[Expression])
  extends Expression with CodegenFallback {
  override def nullable: Boolean = true

  override def eval(input: InternalRow): Any = {
    assert(inputExpressions.length == 1)
    val rst = inputExpressions(1).eval(input).asInstanceOf[Geometry]
    println(rst.toString)
    1
  }

  override def dataType: DataType = IntegerType

  override def children: Seq[Expression] = inputExpressions
}


class MapMatching {
  private var roads: Array[Geometry] = _

  private var points: Column = _

  private val index: RTreeIndex = new RTreeIndex

  private val spark = SparkSession.builder().getOrCreate()

  private def setRoads(roads: Array[Geometry]): Unit = this.roads = roads

  private def setPoints(points: Column): Unit = this.points = points

  private def buildIndex(): Unit = for (road <- roads) index.insert(road.getEnvelopeInternal, road)

  private def broadcastIndex(): Unit = spark.sparkContext.broadcast(index)

  def compute(): Unit = {
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

    val data = Seq(
      Row(1, "POINT (10 20)"),
      Row(2, "LINESTRING (0 0, 10 10, 20 20)"),
      Row(3, "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"),
      Row(4, "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))"),
      Row(5, "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 5 10, 15 5)))"),
      Row(6, null),
      Row(7, "plygon(111, 123)")
    )

    val rdd_d = spark.sparkContext.parallelize(data)
    val schema = StructType(Array(StructField("idx", IntegerType, nullable = false), StructField("wkt", StringType, nullable = true)))

    val df = spark.createDataFrame(rdd_d, schema)

    val broadcast = spark.sparkContext.broadcast()
    val compute = ComputeNearestRoad(Seq(col("idx").expr, lit(GeometryUDT.FromWkt("POINT (20 20)")).expr))
    val rst = df.select(Column{compute})
    rst.show(false)
  }
}
