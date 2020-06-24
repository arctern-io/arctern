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
package org.apache.spark.sql.arctern.index

import java.util

import org.locationtech.jts.geom.Envelope
import org.locationtech.jts.index.strtree.STRtree
import org.locationtech.jts.index.{ItemVisitor, SpatialIndex}

class RTreeIndex extends SpatialIndex with Serializable {
  private var index = new STRtree

  override def insert(itemEnv: Envelope, item: Any): Unit = {
    index.insert(itemEnv, item)
  }

  override def query(searchEnv: Envelope): util.List[_] = {
    index.query(searchEnv)
  }

  override def query(searchEnv: Envelope, visitor: ItemVisitor): Unit = {
    index.query(searchEnv, visitor)
  }

  override def remove(itemEnv: Envelope, item: Any): Boolean = {
    index.remove(itemEnv, item)
  }
}
