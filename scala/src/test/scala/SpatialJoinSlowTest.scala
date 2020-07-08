import org.apache.spark.sql.arctern.SpatialJoin
import org.apache.spark.sql.arctern.functions._
import org.apache.spark.sql.functions._


case class Info(text: String)

class SpatialJoinSlowTest extends AdapterTest {
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

    val fin = SpatialJoin(spark, points, polygons, 'points, 'polygons)()
    1
    fin.show()

    val rst = fin.as[(Long, Long)].collect().toSeq.sorted
    assert(rst.length == 8)
    val ref = Seq(
      (0, 0),
      (1, 0),
      (2, 0),
      (3, 0),
      (4, 0),
      (4, 1),
      (5, 1),
      (6, 2),
    )
    assert(rst == ref)
  }
}
