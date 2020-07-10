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

import org.apache.spark.sql.arctern.{ArcternExpr, CodeGenUtil, GeometryType, GeometryUDT}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.expressions.codegen.Block._
import org.apache.spark.sql.catalyst.expressions.codegen._
import org.apache.spark.sql.catalyst.util.GenericArrayData
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String
import org.apache.zookeeper.KeeperException.UnimplementedException

case class ST_GeomFromText(inputExpr: Seq[Expression]) extends ArcternExpr {

  override def nullable: Boolean = true

  override def eval(input: InternalRow): GenericArrayData = {
    val text = inputExpr(0).eval(input).asInstanceOf[UTF8String].toString
    val geom = GeometryUDT.FromWkt(text)
    GeometryType.serialize(geom)
  }

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {

    val wktExpr = inputExpr.head
    val wktGen = inputExpr.head.genCode(ctx)
    val geoName = ctx.freshName(ev.value)

    val nullSafeEval =
      wktGen.code + ctx.nullSafeExec(wktExpr.nullable, wktGen.isNull) {
        s"""
           |$geoName = ${GeometryUDT.getClass().getName().dropRight(1)}.FromWkt(${wktGen.value}.toString());
           |if ($geoName != null) ${ev.value} = ${CodeGenUtil.serialGeometryCode(geoName)}
       """.stripMargin
      }
    ev.copy(code =
      code"""
          ${CodeGenUtil.mutableGeometryInitCode(geoName)}
          ${CodeGenerator.javaType(ArrayType(ByteType, containsNull = false))} ${ev.value} = ${CodeGenerator.defaultValue(dataType)};
          $nullSafeEval
          boolean ${ev.isNull} = ($geoName == null);
            """)

  }

  override def dataType: DataType = new GeometryUDT

  override def children: Seq[Expression] = inputExpr

  override def inputTypes: Seq[AbstractDataType] = Seq(StringType)

}

case class ST_GeomFromWKB(inputExpr: Seq[Expression]) extends ArcternExpr {

  override def nullable: Boolean = true

  override def eval(input: InternalRow): GenericArrayData = {
    throw new UnimplementedException
  }

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {

    val wktExpr = inputExpr.head
    val wktGen = inputExpr.head.genCode(ctx)
    val geoName = ctx.freshName(ev.value)

    val nullSafeEval =
      wktGen.code + ctx.nullSafeExec(wktExpr.nullable, wktGen.isNull) {
        s"""
           |$geoName = ${GeometryUDT.getClass.getName.dropRight(1)}.FromWkb(${wktGen.value});
           |if ($geoName != null) ${ev.value} = ${CodeGenUtil.serialGeometryCode(geoName)}
       """.stripMargin
      }
    ev.copy(code =
      code"""
          ${CodeGenUtil.mutableGeometryInitCode(geoName)}
          ${CodeGenerator.javaType(ArrayType(ByteType, containsNull = false))} ${ev.value} = ${CodeGenerator.defaultValue(dataType)};
          $nullSafeEval
          boolean ${ev.isNull} = ($geoName == null);
            """)

  }

  override def dataType: DataType = new GeometryUDT

  override def children: Seq[Expression] = inputExpr

  override def inputTypes: Seq[AbstractDataType] = Seq(BinaryType)
}

case class ST_Point(inputExpr: Seq[Expression]) extends ArcternExpr {

  override def nullable: Boolean = true

  override def eval(input: InternalRow): GenericArrayData = {
    throw new UnimplementedException
  }

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {

    val xExpr = inputExpr.head
    val yExpr = inputExpr(1)
    val xGen = inputExpr.head.genCode(ctx)
    val yGen = inputExpr(1).genCode(ctx)
    val geoName = ctx.freshName(ev.value)

    val nullSafeEval =
      xGen.code + ctx.nullSafeExec(xExpr.nullable, xGen.isNull) {
        yGen.code + ctx.nullSafeExec(yExpr.nullable, yGen.isNull) {
          s"""
             |$geoName = new org.locationtech.jts.geom.GeometryFactory().createPoint(new org.locationtech.jts.geom.Coordinate(${xGen.value},${yGen.value}));
             |if ($geoName != null) ${ev.value} = ${CodeGenUtil.serialGeometryCode(geoName)}
          """.stripMargin
        }
      }

    ev.copy(code =
      code"""
          ${CodeGenUtil.mutableGeometryInitCode(geoName)}
          ${CodeGenerator.javaType(ArrayType(ByteType, containsNull = false))} ${ev.value} = ${CodeGenerator.defaultValue(dataType)};
          $nullSafeEval
          boolean ${ev.isNull} = ($geoName == null);
          """)
  }

  override def dataType: DataType = new GeometryUDT

  override def children: Seq[Expression] = inputExpr

  override def inputTypes: Seq[AbstractDataType] = Seq(NumericType, NumericType)
}

case class ST_PolygonFromEnvelope(inputExpr: Seq[Expression]) extends ArcternExpr {

  override def nullable: Boolean = true

  override def eval(input: InternalRow): GenericArrayData = {
    throw new UnimplementedException
  }

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {

    val minXExpr = inputExpr.head
    val minYExpr = inputExpr(1)
    val maxXExpr = inputExpr(2)
    val maxYExpr = inputExpr(3)
    val minXGen = inputExpr.head.genCode(ctx)
    val minYGen = inputExpr(1).genCode(ctx)
    val maxXGen = inputExpr(2).genCode(ctx)
    val maxYGen = inputExpr(3).genCode(ctx)
    val geoName = ctx.freshName(ev.value)

    def coordinateCode(x: ExprCode, y: ExprCode) = {
      s"new org.locationtech.jts.geom.Coordinate(${x.value}, ${y.value});"
    }

    val nullSafeEval =
      minXGen.code + ctx.nullSafeExec(minXExpr.nullable, minXGen.isNull) {
        minYGen.code + ctx.nullSafeExec(minYExpr.nullable, minYGen.isNull) {
          maxXGen.code + ctx.nullSafeExec(maxXExpr.nullable, maxXGen.isNull) {
            maxYGen.code + ctx.nullSafeExec(maxYExpr.nullable, maxYGen.isNull) {
              s"""
                 |org.locationtech.jts.geom.Coordinate[] coordinates = new org.locationtech.jts.geom.Coordinate[5];
                 |coordinates[0] = ${coordinateCode(minXGen, minYGen)}
                 |coordinates[1] = ${coordinateCode(minXGen, maxYGen)}
                 |coordinates[2] = ${coordinateCode(maxXGen, maxYGen)}
                 |coordinates[3] = ${coordinateCode(maxXGen, minYGen)}
                 |coordinates[4] = coordinates[0];
                 |$geoName = new org.locationtech.jts.geom.GeometryFactory().createPolygon(coordinates);
                 |if ($geoName != null) ${ev.value} = ${CodeGenUtil.serialGeometryCode(geoName)}
              """.stripMargin
            }
          }
        }
      }

    ev.copy(code =
      code"""
          ${CodeGenUtil.mutableGeometryInitCode(geoName)}
          ${CodeGenerator.javaType(ArrayType(ByteType, containsNull = false))} ${ev.value} = ${CodeGenerator.defaultValue(dataType)};
          $nullSafeEval
          boolean ${ev.isNull} = ($geoName == null);
          """)
  }

  override def dataType: DataType = new GeometryUDT

  override def children: Seq[Expression] = inputExpr

  override def inputTypes: Seq[AbstractDataType] = Seq(NumericType, NumericType, NumericType, NumericType)
}

case class ST_GeomFromGeoJSON(inputExpr: Seq[Expression]) extends ArcternExpr {

  override def nullable: Boolean = true

  override def eval(input: InternalRow): GenericArrayData = {
    throw new UnimplementedException
  }

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {

    val jsonExpr = inputExpr.head
    val jsonGen = inputExpr.head.genCode(ctx)
    val geoName = ctx.freshName(ev.value)

    val nullSafeEval =
      jsonGen.code + ctx.nullSafeExec(jsonExpr.nullable, jsonGen.isNull) {
        s"""
           |$geoName = ${GeometryUDT.getClass.getName.dropRight(1)}.FromGeoJSON(${jsonGen.value}.toString());
           |if ($geoName != null) ${ev.value} = ${CodeGenUtil.serialGeometryCode(geoName)}
       """.stripMargin
      }
    ev.copy(code =
      code"""
          ${CodeGenUtil.mutableGeometryInitCode(geoName)}
          ${CodeGenerator.javaType(ArrayType(ByteType, containsNull = false))} ${ev.value} = ${CodeGenerator.defaultValue(dataType)};
          $nullSafeEval
          boolean ${ev.isNull} = ($geoName == null);
            """)

  }

  override def dataType: DataType = new GeometryUDT

  override def children: Seq[Expression] = inputExpr

  override def inputTypes: Seq[AbstractDataType] = Seq(StringType)
}

case class ST_AsText(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => CodeGenUtil.utf8StringFromStringCode(s"${GeometryUDT.getClass.getName.dropRight(1)}.ToWkt($geo)"))

  override def dataType: DataType = StringType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_AsWKB(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"${GeometryUDT.getClass.getName.dropRight(1)}.ToWkb($geo)")

  override def dataType: DataType = BinaryType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}

case class ST_AsGeoJSON(inputsExpr: Seq[Expression]) extends ST_UnaryOp {

  override def expr: Expression = inputsExpr.head

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => CodeGenUtil.utf8StringFromStringCode(s"${GeometryUDT.getClass.getName.dropRight(1)}.ToGeoJSON($geo)"))

  override def dataType: DataType = StringType

  override def inputTypes: Seq[AbstractDataType] = Seq(new GeometryUDT)
}
