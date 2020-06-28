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
import org.geotools.geometry.jts.WKTReader2
import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.io.{ParseException, WKBReader, WKBWriter, WKTWriter}
import org.wololo.jts2geojson.{GeoJSONReader, GeoJSONWriter}

class GeometryUDT extends UserDefinedType[Geometry] {
  override def pyUDT: String = "scala_wrapper.GeometryUDT"

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
      case null => null
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
      new WKTReader2().read(obj)
    }
    catch {
      case _: ParseException => null
    }
  }

  def ToGeoJSON(obj: Geometry): String = new GeoJSONWriter().write(obj).toString

  def FromGeoJSON(obj: String): Geometry = {
    try {
      new GeoJSONReader().read(obj)
    }
    catch {
      case _: UnsupportedOperationException => null
    }
  }

  def GetGeoType(obj: Geometry): String = obj.getGeometryType
}

case object GeometryType extends GeometryUDT
