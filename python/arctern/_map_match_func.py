# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=too-many-lines

__all__ = [
    "nearest_location_on_road",
    "nearest_road"
]

from . import arctern_core_

def _to_arrow_array_list(arrow_array):
    if hasattr(arrow_array, 'chunks'):
        return list(arrow_array.chunks)
    return [arrow_array]

def _to_pandas_series(array_list):
    result = None

    for array in array_list:
        if isinstance(array, list):
            for arr in array:
                if result is None:
                    result = arr.to_pandas()
                else:
                    result = result.append(arr.to_pandas(), ignore_index=True)
        else:
            if result is None:
                result = array.to_pandas()
            else:
                result = result.append(array.to_pandas(), ignore_index=True)
    return result

def nearest_location_on_road(roads, gps_points):
    import pyarrow as pa
    arr_roads = pa.array(roads, type='binary')
    arr_gps_points = pa.array(gps_points, type='binary')
    arr_roads = _to_arrow_array_list(arr_roads)
    arr_gps_points = _to_arrow_array_list(arr_gps_points)
    result = arctern_core_.nearest_location_on_road(arr_roads, arr_gps_points)
    return _to_pandas_series(result)

def nearest_road(roads, gps_points):
    import pyarrow as pa
    arr_roads = pa.array(roads, type='binary')
    arr_gps_points = pa.array(gps_points, type='binary')
    arr_roads = _to_arrow_array_list(arr_roads)
    arr_gps_points = _to_arrow_array_list(arr_gps_points)
    result = arctern_core_.nearest_road(arr_roads, arr_gps_points)
    return _to_pandas_series(result)
