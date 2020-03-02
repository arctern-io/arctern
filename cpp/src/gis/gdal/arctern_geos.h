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

#ifndef __cplusplus
#include <stddef.h>
#else
#include <cstddef>
using std::size_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GEOSContextHandle_HS* GEOSContextHandle_t;
class GEOSGeometry;

/************************************************************************
 *
 *  Misc functions
 *
 ***********************************************************************/

/* Return 0 on exception, 1 otherwise */
extern int GEOSHausdorffDistance_r(GEOSContextHandle_t handle, const GEOSGeometry* g1,
                                   const GEOSGeometry* g2, double* dist);

/************************************************************************
 *
 * Memory management
 *
 ***********************************************************************/

extern void GEOSGeom_destroy_r(GEOSContextHandle_t handle, GEOSGeometry* g);

#ifdef __cplusplus
}  // extern "C"
#endif