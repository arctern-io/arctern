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
import org.apache.spark.sql.types.{BooleanType, DataType}
import org.apache.spark.sql.catalyst.expressions.codegen._
import org.apache.spark.sql.catalyst.expressions.codegen.Block._

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

    var leftGeo : String = ""
    var leftGeoDeclare : String = ""
    var leftGeoCode : String = ""
    var rightGeo : String = ""
    var rightGeoDeclare : String = ""
    var rightGeoCode : String = ""

    val leftCode = leftExpr.genCode(ctx)
    val rightCode = rightExpr.genCode(ctx)

    if(CodeGenUtil.isArcternExpr(leftExpr)){
      val (geo, declare, code) = CodeGenUtil.geometryFromArcternExpr(leftCode.code.toString())
      leftGeo = geo; leftGeoDeclare = declare; leftGeoCode = code
    } else {
      val (geo, declare, code) = CodeGenUtil.geometryFromNormalExpr(leftCode)
      leftGeo = geo; leftGeoDeclare = declare; leftGeoCode = code
    }

    if(CodeGenUtil.isArcternExpr(rightExpr)){
      val (geo, declare, code) = CodeGenUtil.geometryFromArcternExpr(rightCode.code.toString())
      rightGeo = geo; rightGeoDeclare = declare; rightGeoCode = code
    } else {
      val (geo, declare, code) = CodeGenUtil.geometryFromNormalExpr(rightCode)
      rightGeo = geo; rightGeoDeclare = declare; rightGeoCode = code
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

    var exprGeo :String = ""
    var exprGeoDeclare :String = ""
    var exprGeoCode :String = ""

    if(CodeGenUtil.isArcternExpr(expr)){
      val (geo, declare, code) = CodeGenUtil.geometryFromArcternExpr(exprCode.code.toString())
      exprGeo = geo; exprGeoDeclare = declare; exprGeoCode = code
    } else {
      val (geo, declare, code) = CodeGenUtil.geometryFromNormalExpr(exprCode)
      exprGeo = geo; exprGeoDeclare = declare; exprGeoCode = code
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
  assert(inputsExpr.length == 2)

  override def leftExpr: Expression = inputsExpr(0)

  override def rightExpr: Expression = inputsExpr(1)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, (left, right) => s"$left.within($right)")

  override def dataType: DataType = BooleanType
}

case class ST_Centroid(inputsExpr: Seq[Expression])extends ST_UnaryOp {
  assert(inputsExpr.length == 1)

  override def expr: Expression = inputsExpr(0)

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = codeGenJob(ctx, ev, geo => s"$geo.getCentroid()")

  override def dataType: DataType = new GeometryUDT

}
