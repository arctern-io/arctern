import org.apache.spark.sql.arctern.functions._
import org.apache.spark.sql.functions._

class IndexedWithinTest extends AdapterTest {
  test("naive") {
    val ss = spark
    import ss.implicits._
    val points_text = Seq(
      "Point(1 1)",
      "Point(1 2)",
      "Point(2 1)",
      "Point(2 2)",
      "Point(4 5)",
      "Point(8 8)",
      "Point(10 10)",
    ).toDF("points_text").withColumn("attr", monotonically_increasing_id())
    val polygons_text = Seq(
      "Polygon((0 0, 3 0, 3 3, 0 3, 0 0))",
      "Polygon((6 6, 3 6, 3 3, 6 3, 6 6))",
      "Polygon((6 6, 9 6, 9 9, 6 9, 6 6))",
    ).toDF("polygons_text").withColumn("id", monotonically_increasing_id())
    val points = points_text.select(st_geomfromtext('points_text).as("points"))
    val polygons = polygons_text.select(st_astext(st_geomfromtext('polygons_text)).as("polygons"))

//    points.show()
    polygons.show()

  }
}
