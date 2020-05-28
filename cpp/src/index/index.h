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

#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <iomanip>

#include "arrow/api.h"
#include "arrow/array.h"

#include <geos/indexStrtree.h>
#include <geos/indexQuadtree.h>
#include <geos/indexBintree.h>
#include <geos/indexChain.h>
#include <geos/indexSweepline.h>
#include <geos/spatialIndex.h>
#include <ogr_geometry.h>

using RTree = GEOS_DLL::geos::index::strtree::STRtree;
using QuadTree = GEOS_DLL::geos::index::quadtree::Quadtree;
using SpatialIndex = GEOS_DLL::geos::index::SpatialIndex;
namespace arctern {
namespace index {

enum class IndexType {
    rTree = 0,
    qTree,
};

class IndexNode {
public:
    IndexNode(): geometry_(nullptr), index_(-1) {};

    IndexNode(OGRGeometry* geo, int32_t index);

    const std::shared_ptr<OGRGeometry>
    geometry() const { return geometry_; }

    const int32_t
    index() const { return index_; }

private:
    std::shared_ptr<OGRGeometry> geometry_;
    int32_t index_;
};



}  // namespace index
}  // namespace arctern