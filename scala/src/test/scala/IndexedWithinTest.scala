import org.apache.spark.sql.arctern.WithinJoin
import org.apache.spark.sql.arctern.functions._
import org.apache.spark.sql.arctern.index.RTreeIndex
import org.apache.spark.sql.functions._
import org.locationtech.jts.geom.{Envelope, Geometry}

import scala.collection.mutable


case class Info(text: String)

class IndexedWithinTest extends AdapterTest {
  test("naive") {
    val ss = spark
    import ss.implicits._
    val points_text = Seq(
      Info("Point(1 1)"),
      Info("Point(1 2)"),
      Info("Point(2 1)"),
      Info("Point(2 2)"),
      Info("Point(3 3)"),
      Info("Point(4 5)"),
      Info("Point(8 8)"),
      Info("Point(10 10)"),
    ).toDF.withColumn("attr", monotonically_increasing_id()).coalesce(3)

    val polygons_text = Seq(
      Info("Polygon((0 0, 3 0, 3.1 3.1, 0 3, 0 0))"),
      Info("Polygon((6 6, 3 6, 2.9 2.9, 6 3, 6 6))"),
      Info("Polygon((6 6, 9 6, 9 9, 6 9, 6 6))"),
    ).toDF.withColumn("id", monotonically_increasing_id()).coalesce(3)

    val points = points_text.select('attr, st_geomfromtext('text).as("points"))

    val polygons = polygons_text.select('id, st_geomfromtext('text).as("polygons"))

    // here feed dog

    val fin = WithinJoin(spark, points, 'attr, 'points, polygons, 'id, 'polygons)()
    fin.show()
    // both are unusable
  }
}
