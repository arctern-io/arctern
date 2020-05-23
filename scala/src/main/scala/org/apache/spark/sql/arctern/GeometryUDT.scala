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
package org.apache.spark.sql.arctern

import org.apache.spark.sql.catalyst.util._
import org.apache.spark.sql.types._
import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.io.{WKBReader, WKBWriter, WKTReader, WKTWriter, ParseException}

class GeometryUDT extends UserDefinedType[Geometry] {
  override def sqlType: DataType = ArrayType(ByteType, containsNull = false)

  override def serialize(obj: Geometry): GenericArrayData = GeometryUDT.GeomSerialize(obj)

  override def deserialize(datum: Any): Geometry = GeometryUDT.GeomDeserialize(datum)

  override def userClass: Class[Geometry] = classOf[Geometry]
}

object GeometryUDT {
  def GeomSerialize(obj: Geometry): GenericArrayData = new GenericArrayData(ToWkb(obj))

  def GeomDeserialize(datum: Any): Geometry = {
    datum match {
      case values: ArrayData => FromWkb(values.toByteArray())
    }
  }

  def ToWkb(obj: Geometry) = new WKBWriter().write(obj)

  def FromWkb(obj: Array[Byte]): Geometry = {
    try {
      new WKBReader().read(obj)
    }
    catch {
      case _: ParseException => null
    }
  }

  def ToWkt(obj: Geometry) = new WKTWriter().write(obj)

  def FromWkt(obj: String): Geometry = {
    try {
      new WKTReader().read(obj)
    }
    catch {
      case _: ParseException => null
    }
  }
}
