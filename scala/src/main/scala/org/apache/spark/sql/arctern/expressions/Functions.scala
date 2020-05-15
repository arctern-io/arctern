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

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.types.{ArrayType, BooleanType, DataType}

case class ST_Within(inputExpr: Seq[Expression]) extends Expression {

  import org.apache.spark.sql.catalyst.expressions.codegen._
  import org.apache.spark.sql.catalyst.expressions.codegen.Block._

  override def nullable: Boolean = false

  override def eval(input: InternalRow): Any = {}

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    val geo0 = inputExpr.head.genCode(ctx)
    val geo1 = inputExpr(1).genCode(ctx)

    ev.copy(code =
      code"""
          ${geo0.code}
          ${geo1.code}

          org.locationtech.jts.geom.Geometry ${ev.value}_geo0 = null;
          org.locationtech.jts.geom.Geometry ${ev.value}_geo1 = null;

          try {
            ${ev.value}_geo0 = new org.locationtech.jts.io.WKBReader().read(${geo0.value}.toByteArray());
            ${ev.value}_geo1 = new org.locationtech.jts.io.WKBReader().read(${geo1.value}.toByteArray());
          } catch(org.locationtech.jts.io.ParseException e) {
            // TODO: add log here
            System.out.println(e.toString());
          }
          boolean[] result = {${ev.value}_geo0.within(${ev.value}_geo1)};
          ${CodeGenerator.javaType(ArrayType(BooleanType, containsNull = false))} ${ev.value} = new org.apache.spark.sql.catalyst.util.GenericArrayData(result);
          """, FalseLiteral)
  }

  override def dataType: DataType = BooleanType

  override def children: Seq[Expression] = inputExpr
}
