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

import org.apache.spark.sql.arctern.GeometryUDT
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.types.{BooleanType, DataType}
import org.apache.spark.sql.catalyst.expressions.codegen._
import org.apache.spark.sql.catalyst.expressions.codegen.Block._

case class ST_Within(inputExpr: Seq[Expression]) extends Expression {

  assert(inputExpr.length == 2)

  override def nullable: Boolean = inputExpr(0).nullable || inputExpr(1).nullable

  override def eval(input: InternalRow): Any = {}

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    val left = inputExpr(0)
    val right = inputExpr(1)

    val leftCode = left.genCode(ctx)
    val rightCode = right.genCode(ctx)

    val resultCode =
      s"""
         |org.locationtech.jts.geom.Geometry ${ev.value}_left = null;
         |org.locationtech.jts.geom.Geometry ${ev.value}_right = null;
         |${ev.value}_left = ${GeometryUDT.getClass().getName().dropRight(1)}.GeomDeserialize(${leftCode.value});
         |${ev.value}_right = ${GeometryUDT.getClass().getName().dropRight(1)}.GeomDeserialize(${rightCode.value});
         |${ev.value} = ${ev.value}_left.within(${ev.value}_right);
         |""".stripMargin

    if (nullable) {
      val nullSafeEval =
        leftCode.code + ctx.nullSafeExec(left.nullable, leftCode.isNull) {
          rightCode.code + ctx.nullSafeExec(right.nullable, rightCode.isNull) {
            s"""
               |${ev.isNull} = false; // resultCode could change nullability.
               |$resultCode
               |""".stripMargin
          }
        }
      ev.copy(code=
        code"""
            boolean ${ev.isNull} = true;
            ${CodeGenerator.javaType(BooleanType)} ${ev.value} = ${CodeGenerator.defaultValue(BooleanType)};
            $nullSafeEval
            """)

    }
    else {
      ev.copy(code =
        code"""
          ${leftCode.code}
          ${rightCode.code}
          ${CodeGenerator.javaType(BooleanType)} ${ev.value} = ${CodeGenerator.defaultValue(BooleanType)};
          $resultCode
          """, FalseLiteral)
    }
  }

  override def dataType: DataType = BooleanType

  override def children: Seq[Expression] = inputExpr
}
