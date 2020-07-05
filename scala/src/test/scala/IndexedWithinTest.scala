import org.apache.spark.sql.arctern.functions._
import org.apache.spark.sql.functions._

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
      Info("Point(4 5)"),
      Info("Point(8 8)"),
      Info("Point(10 10)"),
    ).toDF
//      .withColumn("attr", monotonically_increasing_id())

    val polygons_text = Seq(
      "Polygon((0 0, 3 0, 3 3, 0 3, 0 0))",
      "Polygon((6 6, 3 6, 3 3, 6 3, 6 6))",
      "Polygon((6 6, 9 6, 9 9, 6 9, 6 6))",
    ).toDF("polygons_text").withColumn("id", monotonically_increasing_id())

    val points = points_text.select(st_geomfromtext('text).as("points"))
//    val polygons = polygons_text.select(st_astext(st_geomfromtext('text)).as("polygons_text_again"))

    // both are unusable
    points.show()

//    polygons.show()

  }
}
