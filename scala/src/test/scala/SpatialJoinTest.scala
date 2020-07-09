import org.apache.spark.sql.arctern.functions._
import org.apache.spark.sql.arctern.{ContainsOp, GeometryUDT, SpatialJoin}
import org.apache.spark.sql.catalyst.plans.{FullOuter, Inner, LeftOuter, RightOuter}
import org.apache.spark.sql.functions._
import org.locationtech.jts.geom.Geometry


case class Info(text: String)

class SpatialJoinTest extends AdapterTest {
  lazy val ss = spark

  import ss.implicits._

  implicit def regularize(dataset: Seq[Info]): Seq[Info] = {
    dataset.map { case Info(text) =>
      val geom = GeometryUDT.FromWkt(text)
      Info(geom.toString)
    }
  }

  lazy val points_dataset = regularize(Seq(
    Info("Point(1 1)"),
    Info("Point(1 2)"),
    Info("Point(2 1)"),
    Info("Point(2 2)"),
    Info("Point(3 3)"),
    Info("Point(4 5)"),
    Info("Point(8 8)"),
    Info("Point(10 10)"),
  ))
  lazy val polygons_dataset = regularize(Seq(
    Info("Polygon((0 0, 3 0, 3.1 3.1, 0 3, 0 0))"),
    Info("Polygon((6 6, 3 6, 2.9 2.9, 6 3, 6 6))"),
    Info("Polygon((6 6, 9 6, 9 9, 6 9, 6 6))"),
    Info("Polygon((100 100, 100 101, 101 101, 101 100, 100 100))"),
  ))

  def get_points_text() = {
    val points_text = points_dataset.toDF
      .withColumn("attr", monotonically_increasing_id())
      .withColumn("dup", lit(10) * monotonically_increasing_id())
      .coalesce(3)
    points_text
  }

  def get_polygons_text() = {
    val polygons_text = polygons_dataset.toDF
      .withColumn("id", monotonically_increasing_id())
      .withColumn("dup", lit(10) * monotonically_increasing_id())
      .coalesce(3)
    polygons_text
  }

  def assertVerify(dataset: Seq[Info])(input: Seq[(Option[Long], Option[Long], Geometry)]): Unit = {
    input.foreach {
      case (id, dup, geom) if id.isEmpty || dup.isEmpty => {
        assert(id.isEmpty)
        assert(dup.isEmpty)
        assert(geom == null)
      }
      case (Some(id), Some(dup), geom) => {
        assert(dup == id * 10)
        assert(geom.toString == dataset(id.toInt).text)
      }
      case _ => throw new RuntimeException
    }
  }

  test("right-within") {
    val points = get_points_text().select('attr, 'dup, st_geomfromtext('text).as("points"))
    val polygons = get_polygons_text().select('id, 'dup, st_geomfromtext('text).as("polygons"))
    val fin = SpatialJoin(spark, points, polygons, "points", "polygons", RightOuter)
    val ref_columns = Seq("attr", "dup_left", "points", "id", "dup_right", "polygons")
    assert(fin.columns.toSeq == ref_columns)

    fin.show()
    val rst = fin.as[(Option[Long], Option[Long], Geometry, Option[Long], Option[Long], Geometry)]
      .collect().toSeq.sortBy(tp => (tp._1, tp._4))
    assertVerify(points_dataset)(rst.map(tp => (tp._1, tp._2, tp._3)))
    assertVerify(polygons_dataset)(rst.map(tp => (tp._4, tp._5, tp._6)))

    val ref = Seq[(Option[Long], Option[Long])](
      (None, Some(3)),
      (Some(0), Some(0)),
      (Some(1), Some(0)),
      (Some(2), Some(0)),
      (Some(3), Some(0)),
      (Some(4), Some(0)),
      (Some(4), Some(1)),
      (Some(5), Some(1)),
      (Some(6), Some(2)),
    ).sorted
    assert(rst.map(tp => (tp._1, tp._4)) == ref)
  }

  test("left-within") {
    val points = get_points_text().select('attr, 'dup, st_geomfromtext('text).as("points"))
    val polygons = get_polygons_text().select('id, 'dup, st_geomfromtext('text).as("polygons"))
    val fin = SpatialJoin(spark, points, polygons, "points", "polygons", LeftOuter)
    val ref_columns = Seq("attr", "dup_left", "points", "id", "dup_right", "polygons")
    assert(fin.columns.toSeq == ref_columns)

    fin.show()
    val rst = fin.as[(Option[Long], Option[Long], Geometry, Option[Long], Option[Long], Geometry)]
      .collect().toSeq.sortBy(tp => (tp._1, tp._4))
    assertVerify(points_dataset)(rst.map(tp => (tp._1, tp._2, tp._3)))
    assertVerify(polygons_dataset)(rst.map(tp => (tp._4, tp._5, tp._6)))

    val ref = Seq[(Option[Long], Option[Long])](
      (Some(7), None),
      (Some(0), Some(0)),
      (Some(1), Some(0)),
      (Some(2), Some(0)),
      (Some(3), Some(0)),
      (Some(4), Some(0)),
      (Some(4), Some(1)),
      (Some(5), Some(1)),
      (Some(6), Some(2)),
    ).sorted
    assert(rst.map(tp => (tp._1, tp._4)) == ref)
  }

  test("full-within") {
    val points = get_points_text().select('attr, 'dup, st_geomfromtext('text).as("points"))
    val polygons = get_polygons_text().select('id, 'dup, st_geomfromtext('text).as("polygons"))
    val fin = SpatialJoin(spark, points, polygons, "points", "polygons", FullOuter)
    val ref_columns = Seq("attr", "dup_left", "points", "id", "dup_right", "polygons")
    assert(fin.columns.toSeq == ref_columns)

    fin.show()
    val rst = fin.as[(Option[Long], Option[Long], Geometry, Option[Long], Option[Long], Geometry)]
      .collect().toSeq.sortBy(tp => (tp._1, tp._4))
    assertVerify(points_dataset)(rst.map(tp => (tp._1, tp._2, tp._3)))
    assertVerify(polygons_dataset)(rst.map(tp => (tp._4, tp._5, tp._6)))

    val ref = Seq[(Option[Long], Option[Long])](
      (None, Some(3)),
      (Some(7), None),
      (Some(0), Some(0)),
      (Some(1), Some(0)),
      (Some(2), Some(0)),
      (Some(3), Some(0)),
      (Some(4), Some(0)),
      (Some(4), Some(1)),
      (Some(5), Some(1)),
      (Some(6), Some(2)),
    ).sorted
    assert(rst.map(tp => (tp._1, tp._4)) == ref)
  }

  test("inner-within") {
    val points = get_points_text().select('attr, 'dup, st_geomfromtext('text).as("points"))
    val polygons = get_polygons_text().select('id, 'dup, st_geomfromtext('text).as("polygons"))
    val fin = SpatialJoin(spark, points, polygons, "points", "polygons")
    val ref_columns = Seq("attr", "dup_left", "points", "id", "dup_right", "polygons")
    assert(fin.columns.toSeq == ref_columns)

    fin.show()
    val rst = fin.as[(Option[Long], Option[Long], Geometry, Option[Long], Option[Long], Geometry)]
      .collect().toSeq.sortBy(tp => (tp._1, tp._4))
    assertVerify(points_dataset)(rst.map(tp => (tp._1, tp._2, tp._3)))
    assertVerify(polygons_dataset)(rst.map(tp => (tp._4, tp._5, tp._6)))

    val ref = Seq[(Option[Long], Option[Long])](
      (Some(0), Some(0)),
      (Some(1), Some(0)),
      (Some(2), Some(0)),
      (Some(3), Some(0)),
      (Some(4), Some(0)),
      (Some(4), Some(1)),
      (Some(5), Some(1)),
      (Some(6), Some(2)),
    ).sorted
    assert(rst.map(tp => (tp._1, tp._4)) == ref)
  }

  test("inner-contains") {
    val polygons = get_polygons_text().select('id, 'dup, st_geomfromtext('text).as("polygons"))
    val points = get_points_text().select('attr, 'dup, st_geomfromtext('text).as("points"))
    val fin = SpatialJoin(spark, polygons, points, "polygons", "points", Inner, ContainsOp)
    val ref_columns = Seq("id", "dup_left", "polygons", "attr", "dup_right", "points")
    assert(fin.columns.toSeq == ref_columns)

    fin.show()
    val rst = fin.as[(Option[Long], Option[Long], Geometry, Option[Long], Option[Long], Geometry)]
      .collect().toSeq.sortBy(tp => (tp._1, tp._4))

    assertVerify(polygons_dataset)(rst.map(tp => (tp._1, tp._2, tp._3)))
    assertVerify(points_dataset)(rst.map(tp => (tp._4, tp._5, tp._6)))

    val ref = Seq[(Option[Long], Option[Long])](
      (Some(0), Some(0)),
      (Some(0), Some(1)),
      (Some(0), Some(2)),
      (Some(0), Some(3)),
      (Some(0), Some(4)),
      (Some(1), Some(4)),
      (Some(1), Some(5)),
      (Some(2), Some(6)),
    ).sorted
    assert(rst.map(tp => (tp._1, tp._4)) == ref)
  }

  test("full-contains") {
    val polygons = get_polygons_text().select('id, 'dup, st_geomfromtext('text).as("polygons"))
    val points = get_points_text().select('attr, 'dup, st_geomfromtext('text).as("points"))
    val fin = SpatialJoin(spark, polygons, points, "polygons", "points", FullOuter, ContainsOp)
    val ref_columns = Seq("id", "dup_left", "polygons", "attr", "dup_right", "points")
    assert(fin.columns.toSeq == ref_columns)

    fin.show()
    val rst = fin.as[(Option[Long], Option[Long], Geometry, Option[Long], Option[Long], Geometry)]
      .collect().toSeq.sortBy(tp => (tp._1, tp._4))

    assertVerify(polygons_dataset)(rst.map(tp => (tp._1, tp._2, tp._3)))
    assertVerify(points_dataset)(rst.map(tp => (tp._4, tp._5, tp._6)))

    val ref = Seq[(Option[Long], Option[Long])](
      (Some(3), None),
      (None, Some(7)),
      (Some(0), Some(0)),
      (Some(0), Some(1)),
      (Some(0), Some(2)),
      (Some(0), Some(3)),
      (Some(0), Some(4)),
      (Some(1), Some(4)),
      (Some(1), Some(5)),
      (Some(2), Some(6)),
    ).sorted
    assert(rst.map(tp => (tp._1, tp._4)) == ref)
  }

  test("left-contains") {
    val polygons = get_polygons_text().select('id, 'dup, st_geomfromtext('text).as("polygons"))
    val points = get_points_text().select('attr, 'dup, st_geomfromtext('text).as("points"))
    val fin = SpatialJoin(spark, polygons, points, "polygons", "points", LeftOuter, ContainsOp)
    val ref_columns = Seq("id", "dup_left", "polygons", "attr", "dup_right", "points")
    assert(fin.columns.toSeq == ref_columns)

    fin.show()
    val rst = fin.as[(Option[Long], Option[Long], Geometry, Option[Long], Option[Long], Geometry)]
      .collect().toSeq.sortBy(tp => (tp._1, tp._4))

    assertVerify(polygons_dataset)(rst.map(tp => (tp._1, tp._2, tp._3)))
    assertVerify(points_dataset)(rst.map(tp => (tp._4, tp._5, tp._6)))

    val ref = Seq[(Option[Long], Option[Long])](
      (Some(3), None),
      (Some(0), Some(0)),
      (Some(0), Some(1)),
      (Some(0), Some(2)),
      (Some(0), Some(3)),
      (Some(0), Some(4)),
      (Some(1), Some(4)),
      (Some(1), Some(5)),
      (Some(2), Some(6)),
    ).sorted
    assert(rst.map(tp => (tp._1, tp._4)) == ref)
  }

  test("right-contains") {
    val polygons = get_polygons_text().select('id, 'dup, st_geomfromtext('text).as("polygons"))
    val points = get_points_text().select('attr, 'dup, st_geomfromtext('text).as("points"))
    val fin = SpatialJoin(spark, polygons, points, "polygons", "points", RightOuter, ContainsOp)
    val ref_columns = Seq("id", "dup_left", "polygons", "attr", "dup_right", "points")
    assert(fin.columns.toSeq == ref_columns)

    fin.show()
    val rst = fin.as[(Option[Long], Option[Long], Geometry, Option[Long], Option[Long], Geometry)]
      .collect().toSeq.sortBy(tp => (tp._1, tp._4))

    assertVerify(polygons_dataset)(rst.map(tp => (tp._1, tp._2, tp._3)))
    assertVerify(points_dataset)(rst.map(tp => (tp._4, tp._5, tp._6)))

    val ref = Seq[(Option[Long], Option[Long])](
      (None, Some(7)),
      (Some(0), Some(0)),
      (Some(0), Some(1)),
      (Some(0), Some(2)),
      (Some(0), Some(3)),
      (Some(0), Some(4)),
      (Some(1), Some(4)),
      (Some(1), Some(5)),
      (Some(2), Some(6)),
    ).sorted
    assert(rst.map(tp => (tp._1, tp._4)) == ref)
  }
  test("geom-rename") {
    val polygons = get_polygons_text().select('id, 'dup, st_geomfromtext('text).as("geom"))
    val points = get_points_text().select('attr, 'dup, st_geomfromtext('text).as("geom"))
    val fin = SpatialJoin(spark, polygons, points, "geom", "geom", RightOuter, ContainsOp)
    val ref_columns = Seq("id", "dup_left", "geom_left", "attr", "dup_right", "geom_right")
    assert(fin.columns.toSeq == ref_columns)
  }
}
