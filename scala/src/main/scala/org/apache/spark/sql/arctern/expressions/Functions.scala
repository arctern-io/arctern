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
package org.apache.spark.sql.arctern.expressions

import org.apache.spark.sql.arctern.{ArcternExpr, CodeGenUtil, GeometryUDT}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.expressions.codegen.Block._
import org.apache.spark.sql.catalyst.expressions.codegen._
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.catalyst.util.ArrayData._
import org.apache.spark.sql.types._
import org.geotools.geometry.jts.JTS
import org.geotools.referencing.CRS
import org.locationtech.jts.geom.{Geometry, GeometryFactory, MultiPolygon, Polygon}

object utils {
  def envelopeAsList(geom: Geometry): ArrayData = {
    if (geom == null || geom.isEmpty) {
      val negInf = scala.Double.NegativeInfinity
      val posInf = scala.Double.PositiveInfinity
      toArrayData(Array(posInf, posInf, negInf, negInf))
    } else {
      val env = geom.getEnvelopeInternal
      val arr = Array(env.getMinX, env.getMinY, env.getMaxX, env.getMaxY)
      toArrayData(arr)
    }
  }

  def distanceSphere(from: Geometry, to: Geometry): Double = {
    var distance = -1.0
    if (!from.getGeometryType.equals("Point") || !to.getGeometryType.equals("Point")) {
      distance
    } else {
      // get coordinates
      val fromlon = from.getInteriorPoint.getX
      val fromlat = from.getInteriorPoint.getY
      val tolon = to.getInteriorPoint.getX
      val tolat = to.getInteriorPoint.getY
      if ((fromlat > 180) || (fromlat < -180) || (fromlon > 90) || (fromlon < -90) ||
        (tolat > 180) || (tolat < -180) || (tolon > 90) || (tolon < -90)) {
        distance
      } else {
        // calculate distance
        val latitudeArc = (fromlat - tolat) * 0.017453292519943295769236907684886
        val longitudeArc = (fromlon - tolon) * 0.017453292519943295769236907684886
        var latitudeH = java.lang.Math.sin(latitudeArc * 0.5)
        latitudeH *= latitudeH
        var lontitudeH = java.lang.Math.sin(longitudeArc * 0.5)
        lontitudeH *= lontitudeH
        val tmp = java.lang.Math.cos(fromlat * 0.017453292519943295769236907684886) *
          java.lang.Math.cos(tolat * 0.017453292519943295769236907684886)
        distance = 6372797.560856 * (2.0 * java.lang.Math.asin(java.lang.Math.sqrt(latitudeH + tmp * lontitudeH)))
        distance
      }
    }
  }

  def transform(geo: Geometry, sourceCRS: String, targetCRS: String): Geometry = {
    System.setProperty("org.geotools.referencing.forceXY", "true")
    val sourceCRScode = CRS.decode(sourceCRS)
    val targetCRScode = CRS.decode(targetCRS)
    val transform = CRS.findMathTransform(sourceCRScode, targetCRScode)
    val res = JTS.transform(geo, transform)
    res
  }

  def makeValid(geo: Geometry): Geometry = {
    val geoType = geo.getGeometryType
    if (geoType != "Polygon" || geoType != "MultiPolygon") return geo
    val polygonList: java.util.List[Polygon] = geo match {
      case g: Polygon =>
        JTS.makeValid(g, true)
      case g: MultiPolygon =>
        val list = new java.util.ArrayList[Polygon]
        for (i <- 0 until g.getNumGeometries) {
          val polygon = g.getGeometryN(i).asInstanceOf[Polygon]
          val polygons = JTS.makeValid(polygon, true)
          for (j <- 0 until polygons.size()) list.add(polygons.get(j))
        }
        list
      case _ => null
    }
    val result = polygonList.toArray.map(g => {
      val polygon = g.asInstanceOf[Geometry]
      polygon
    })
    val res = if (result.length == 1) result(0) else new GeometryFactory().createGeometryCollection(result)
    res
  }

}

abstract class ST_BinaryOp extends ArcternExpr {

  def leftExpr: Expression

  def rightExpr: Expression

  override def nullable: Boolean = leftExpr.nullable || rightExpr.nullable

  override def eval(input: InternalRow): Any = {
    throw new RuntimeException("call implement method")
  }

  override def children: Seq[Expression] = Seq(leftExpr, rightExpr)

  protected def codeGenJob(ctx: CodegenContext, ev: ExprCode, f: (String, String) => String): ExprCode = {
    assert(CodeGenUtil.isGeometryExpr(leftExpr))
    assert(CodeGenUtil.isGeometryExpr(rightExpr))

    var leftGeo: String = ""
    var leftGeoDeclare: String = ""
    var leftGeoCode: String = ""
    var rightGeo: String = ""
    var rightGeoDeclare: String = ""
    var rightGeoCode: String = ""

    val leftCode = leftExpr.genCode(ctx)
    val rightCode = rightExpr.genCode(ctx)

    if (CodeGenUtil.isArcternExpr(leftExpr)) {
      val (geo, declare, code) = CodeGenUtil.geometryFromArcternExpr(leftCode.code.toString())
      leftGeo = geo;
      leftGeoDeclare = declare;
      leftGeoCode = code
    } else {
      val (geo, declare, code) = CodeGenUtil.geometryFromNormalExpr(leftCode)
      leftGeo = geo;
      leftGeoDeclare = declare;
      leftGeoCode = code
    }

    if (CodeGenUtil.isArcternExpr(rightExpr)) {
      val (geo, declare, code) = CodeGenUtil.geometryFromArcternExpr(rightCode.code.toString())
      rightGeo = geo;
      rightGeoDeclare = declare;
      rightGeoCode = code
    } else {
      val (geo, declare, code) = CodeGenUtil.geometryFromNormalExpr(rightCode)
      rightGeo = geo;
      rightGeoDeclare = declare;
      rightGeoCode = code
    }

    val assignment = CodeGenUtil.assignmentCode(f(leftGeo, rightGeo), ev.value, dataType)

    if (nullable) {
      val nullSafeEval =
        leftGeoCode + ctx.nullSafeExec(leftExpr.nullable, leftCode.isNull) {
          rightGeoCode + ctx.nullSafeExec(rightExpr.nullable, rightCode.isNull) {
            s"""
               |${ev.isNull} = false; // resultCode could change nullability.
               |$assignment
               |""".stripMargin
          }
        }
      ev.copy(code =
        code"""
            boolean ${ev.isNull} = true;
            $leftGeoDeclare
            $rightGeoDeclare
            ${CodeGenerator.javaType(dataType)} ${ev.value} = ${CodeGenerator.defaultValue(dataType)};
            $nullSafeEval
            """)

    }
    else {
      ev.copy(code =
        code"""
          $leftGeoDeclare
          $rightGeoDeclare
          $leftGeoCode
          $rightGeoCode
          ${CodeGenerator.javaType(dataType)} ${ev.value} = ${CodeGenerator.defaultValue(dataType)};
          $assignment
          """, FalseLiteral)
    }
  }

}

abstract class ST_UnaryOp extends ArcternExpr {

  def expr: Expression

  override def nullable: Boolean = expr.nullable

  override def eval(input: InternalRow): Any = {
    throw new RuntimeException("call implement method")
  }

  override def children: Seq[Expression] = Seq(expr)

  protected def codeGenJob(ctx: CodegenContext, ev: ExprCode, f: String => String): ExprCode = {
    assert(CodeGenUtil.isGeometryExpr(expr))

    val exprCode = expr.genCode(ctx)

    var exprGeo: String = ""
    var exprGeoDeclare: String = ""
    var exprGeoCode: String = ""

    if (CodeGenUtil.isArcternExpr(expr)) {
      val (geo, declare, code) = CodeGenUtil.geometryFromArcternExpr(exprCode.code.toString())
      exprGeo = geo;
      exprGeoDeclare = declare;
      exprGeoCode = code
    } else {
      val (geo, declare, code) = CodeGenUtil.geometryFromNormalExpr(exprCode)
      exprGeo = geo;
      exprGeoDeclare = declare;
      exprGeoCode = code
    }

    val assignment = CodeGenUtil.assignmentCode(f(exprGeo), ev.value, dataType)

    if (nullable) {
      val nullSafeEval =
        exprGeoCode + ctx.nullSafeExec(expr.nullable, exprCode.isNull) {
          s"""
             |${ev.isNull} = false; // resultCode could change nullability.
             |$assignment
             |""".stripMargin
        }
      ev.copy(code =
        code"""
            boolean ${ev.isNull} = true;
            $exprGeoDeclare
            ${CodeGenerator.javaType(dataType)} ${ev.value} = ${CodeGenerator.defaultValue(dataType)};
            $nullSafeEval
            """)
    } else {
      ev.copy(code =
        code"""
            $exprGeoDeclare
            $exprGeoCode
            ${CodeGenerator.javaType(dataType)} ${ev.value} = ${CodeGenerator.defaultValue(dataType)};
            $assignment
            """, FalseLiteral)
    }
  }

}


case class ST_Within(inputsExpr: Seq[Expression]) extends ST_BinaryOp {

  override def leftExpr: Expression = inputsExpr(0)

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"$left.within($right)")

  override def dataType: DataType = BooleanType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, new GeometryUDT)
}

case class ST_Centroid(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr(0)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"$geo.getCentroid()")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_IsValid(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"new org.locationtech.jts.operation.valid.IsValidOp($geo).isValid()")

  override def dataType: DataType = BooleanType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_GeometryType(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => CodeGenUtil.utf8StringFromStringCode(s"${GeometryUDT.getClass.getName.dropRight(1)}.GetGeoType($geo)"))

  override def dataType: DataType = StringType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_IsSimple(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"$geo.isSimple()")

  override def dataType: DataType = BooleanType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_NPoints(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"$geo.getNumPoints()")

  override def dataType: DataType = IntegerType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_Envelope(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"$geo.getEnvelope()")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_Buffer(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  def distanceValue(ctx: CodegenContext): ExprValue = inputsExpr(1).genCode(ctx).value

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"$geo.buffer(${distanceValue(ctx)})")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, NumericType)

  override def children: Seq[Expression] = inputsExpr
}

case class ST_PrecisionReduce(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  def precisionValue(ctx: CodegenContext): ExprValue = inputsExpr(1).genCode(ctx).value

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"org.locationtech.jts.precision.GeometryPrecisionReducer.reduce($geo, new org.locationtech.jts.geom.PrecisionModel(${precisionValue(ctx)}))")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, NumericType)

  override def children: Seq[Expression] = inputsExpr
}

case class ST_Intersection(inputsExpr: Seq[Expression]) extends ST_BinaryOp {

  override def leftExpr: Expression = inputsExpr.head

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"$left.intersection($right)")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, new GeometryUDT)
}

case class ST_SimplifyPreserveTopology(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  def toleranceValue(ctx: CodegenContext): ExprValue = inputsExpr(1).genCode(ctx).value

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"org.locationtech.jts.simplify.TopologyPreservingSimplifier.simplify($geo, ${toleranceValue(ctx)})")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, NumericType)

  override def children: Seq[Expression] = inputsExpr
}

case class ST_ConvexHull(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"$geo.convexHull()")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_Area(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"$geo.getArea()")

  override def dataType: DataType = DoubleType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_Length(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"$geo.getLength()")

  override def dataType: DataType = DoubleType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_HausdorffDistance(inputsExpr: Seq[Expression]) extends ST_BinaryOp {

  override def leftExpr: Expression = inputsExpr.head

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"org.locationtech.jts.algorithm.distance.DiscreteHausdorffDistance.distance($left, $right)")

  override def dataType: DataType = DoubleType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, new GeometryUDT)
}

case class ST_Distance(inputsExpr: Seq[Expression]) extends ST_BinaryOp {

  override def leftExpr: Expression = inputsExpr.head

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"$left.distance($right)")

  override def dataType: DataType = DoubleType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, new GeometryUDT)
}

case class ST_Equals(inputsExpr: Seq[Expression]) extends ST_BinaryOp {

  override def leftExpr: Expression = inputsExpr.head

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"$left.equals($right)")

  override def dataType: DataType = BooleanType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, new GeometryUDT)
}

case class ST_Touches(inputsExpr: Seq[Expression]) extends ST_BinaryOp {

  override def leftExpr: Expression = inputsExpr.head

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"$left.touches($right)")

  override def dataType: DataType = BooleanType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, new GeometryUDT)
}

case class ST_Overlaps(inputsExpr: Seq[Expression]) extends ST_BinaryOp {

  override def leftExpr: Expression = inputsExpr.head

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"$left.overlaps($right)")

  override def dataType: DataType = BooleanType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, new GeometryUDT)
}

case class ST_Crosses(inputsExpr: Seq[Expression]) extends ST_BinaryOp {

  override def leftExpr: Expression = inputsExpr.head

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"$left.crosses($right)")

  override def dataType: DataType = BooleanType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, new GeometryUDT)
}

case class ST_Contains(inputsExpr: Seq[Expression]) extends ST_BinaryOp {

  override def leftExpr: Expression = inputsExpr.head

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"$left.contains($right)")

  override def dataType: DataType = BooleanType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, new GeometryUDT)
}

case class ST_Intersects(inputsExpr: Seq[Expression]) extends ST_BinaryOp {

  override def leftExpr: Expression = inputsExpr.head

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"$left.intersects($right)")

  override def dataType: DataType = BooleanType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, new GeometryUDT)
}

case class ST_DistanceSphere(inputsExpr: Seq[Expression]) extends ST_BinaryOp {

  override def leftExpr: Expression = inputsExpr.head

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"org.apache.spark.sql.arctern.expressions.utils.distanceSphere($left, $right)")

  override def dataType: DataType = DoubleType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, new GeometryUDT)
}

case class ST_Transform(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  def sourceCRSCode(ctx: CodegenContext): ExprValue = inputsExpr(1).genCode(ctx).value

  def targetCRSCode(ctx: CodegenContext): ExprValue = inputsExpr(2).genCode(ctx).value

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"org.apache.spark.sql.arctern.expressions.utils.transform($geo, ${sourceCRSCode(ctx)}.toString(), ${targetCRSCode(ctx)}.toString())")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, StringType, StringType)
}

case class ST_MakeValid(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"org.apache.spark.sql.arctern.expressions.utils.makeValid($geo)")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_CurveToLine(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"$geo")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_Translate(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  def shifterXValue(ctx: CodegenContext): ExprValue = inputsExpr(1).genCode(ctx).value

  def shifterYValue(ctx: CodegenContext): ExprValue = inputsExpr(2).genCode(ctx).value

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"new org.locationtech.jts.geom.util.AffineTransformation().translate(${shifterXValue(ctx)}, ${shifterYValue(ctx)}).transform($geo)")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, NumericType, NumericType)

  override def children: Seq[Expression] = inputsExpr
}

case class ST_Rotate(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  def rotationAngle(ctx: CodegenContext): ExprValue = inputsExpr(1).genCode(ctx).value

  def rotateX(ctx: CodegenContext): ExprValue = inputsExpr(2).genCode(ctx).value

  def rotateY(ctx: CodegenContext): ExprValue = inputsExpr(3).genCode(ctx).value

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"new org.locationtech.jts.geom.util.AffineTransformation().rotate(${rotationAngle(ctx)}, ${rotateX(ctx)}, ${rotateY(ctx)}).transform($geo)")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, NumericType, NumericType, NumericType)

  override def children: Seq[Expression] = inputsExpr
}

case class ST_SymDifference(inputsExpr: Seq[Expression]) extends ST_BinaryOp {

  override def leftExpr: Expression = inputsExpr.head

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"$left.symDifference($right)")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, new GeometryUDT)
}

case class ST_Difference(inputsExpr: Seq[Expression]) extends ST_BinaryOp {

  override def leftExpr: Expression = inputsExpr.head

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"$left.difference($right)")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, new GeometryUDT)
}

case class ST_Union(inputsExpr: Seq[Expression]) extends ST_BinaryOp {

  override def leftExpr: Expression = inputsExpr.head

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"$left.union($right)")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, new GeometryUDT)
}

case class ST_Disjoint(inputsExpr: Seq[Expression]) extends ST_BinaryOp {

  override def leftExpr: Expression = inputsExpr.head

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"$left.disjoint($right)")

  override def dataType: DataType = BooleanType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, new GeometryUDT)
}

case class ST_IsEmpty(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"$geo.isEmpty()")

  override def dataType: DataType = BooleanType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_Boundary(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"$geo.getBoundary()")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_ExteriorRing(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"""$geo.getGeometryType().equals("Polygon") ? new org.locationtech.jts.geom.GeometryFactory().createPolygon($geo.getCoordinates()).getExteriorRing() : $geo """)

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_Scale(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  def factorX(ctx: CodegenContext): ExprValue = inputsExpr(1).genCode(ctx).value

  def factorY(ctx: CodegenContext): ExprValue = inputsExpr(2).genCode(ctx).value

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"new org.locationtech.jts.geom.util.AffineTransformation().scale(${factorX(ctx)}, ${factorY(ctx)}).transform($geo)")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, NumericType, NumericType)

  override def children: Seq[Expression] = inputsExpr
}

case class ST_Affine(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  def a(ctx: CodegenContext): ExprValue = inputsExpr(1).genCode(ctx).value

  def b(ctx: CodegenContext): ExprValue = inputsExpr(2).genCode(ctx).value

  def d(ctx: CodegenContext): ExprValue = inputsExpr(3).genCode(ctx).value

  def e(ctx: CodegenContext): ExprValue = inputsExpr(4).genCode(ctx).value

  def offsetX(ctx: CodegenContext): ExprValue = inputsExpr(5).genCode(ctx).value

  def offsetY(ctx: CodegenContext): ExprValue = inputsExpr(6).genCode(ctx).value

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"new org.locationtech.jts.geom.util.AffineTransformation(${a(ctx)}, ${b(ctx)}, ${d(ctx)}, ${e(ctx)}, ${offsetX(ctx)}, ${offsetY(ctx)}).transform($geo)")

  override def dataType: DataType = new GeometryUDT

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT, NumericType, NumericType, NumericType, NumericType, NumericType, NumericType)

  override def children: Seq[Expression] = inputsExpr
}
