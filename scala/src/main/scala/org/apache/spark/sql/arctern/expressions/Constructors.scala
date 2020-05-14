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
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}

case class ST_GeomFromText(inputExpr: Seq[Expression]) extends Expression {

  import org.apache.spark.sql.catalyst.expressions.codegen._
  import org.apache.spark.sql.catalyst.expressions.codegen.Block._

  override def nullable: Boolean = false

  override def eval(input: InternalRow): Any = {}

  override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    val wkt = inputExpr.head.genCode(ctx)

    ev.copy(code =
      code"""
          ${wkt.code}

          org.locationtech.jts.geom.Geometry ${ev.value}_geo = null;
          try {
            ${ev.value}_geo = new org.locationtech.jts.io.WKTReader().read(${wkt.value}.toString());
          } catch(org.locationtech.jts.io.ParseException e) {
            // TODO: add log here
            System.out.println(e.toString());
          }
          org.apache.spark.sql.arctern.GeometryUDT ${ev.value}_geo_udt = new org.apache.spark.sql.arctern.GeometryUDT();
          ${CodeGenerator.javaType(ArrayType(StringType, containsNull = false))} ${ev.value} = ${ev.value}_geo_udt.serialize(${ev.value}_geo);
          """, FalseLiteral)
  }

  override def dataType: DataType = new GeometryUDT

  override def children: Seq[Expression] = inputExpr
}
