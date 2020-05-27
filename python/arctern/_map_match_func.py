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
    "snap_to_road"
]

from . import arctern_core_

def arctern_udf(*arg_types):
    def decorate(func):
        from functools import wraps

        @wraps(func)
        def wrapper(*warpper_args):
            import pandas as pd
            pd_series_type = type(pd.Series([None]))
            array_len = 1
            for arg in warpper_args:
                if isinstance(arg, pd_series_type):
                    array_len = len(arg)
                    break
            func_args = []
            func_arg_idx = 0
            for arg_type in arg_types:
                if arg_type is None:
                    func_args.append(warpper_args[func_arg_idx])
                else:
                    assert isinstance(arg_type, str)
                    if len(arg_type) == 0:
                        func_args.append(warpper_args[func_arg_idx])
                    elif isinstance(warpper_args[func_arg_idx], pd_series_type):
                        assert len(warpper_args[func_arg_idx]) == array_len
                        func_args.append(warpper_args[func_arg_idx])
                    else:
                        if arg_type == 'binary':
                            arg_type = 'object'
                        arg = pd.Series([warpper_args[func_arg_idx] for _ in range(array_len)], dtype=arg_type)
                        func_args.append(arg)
                func_arg_idx = func_arg_idx + 1
            while func_arg_idx < len(warpper_args):
                func_args.append(warpper_args[func_arg_idx])
                func_arg_idx = func_arg_idx + 1
            return func(*func_args)
        return wrapper
    return decorate

def arctern_caller(func, *func_args):
    import pyarrow
    num_chunks = 1
    for arg in func_args:
        # pylint: disable=c-extension-no-member
        if isinstance(arg, pyarrow.lib.ChunkedArray):
            num_chunks = len(arg.chunks)
            break

    if num_chunks <= 1:
        result = func(*func_args)
        return result.to_pandas()

    result_total = None
    for chunk_idx in range(num_chunks):
        args = []
        for arg in func_args:
            # pylint: disable=c-extension-no-member
            if isinstance(arg, pyarrow.lib.ChunkedArray):
                args.append(arg.chunks[chunk_idx])
            else:
                args.append(arg)
        result = func(*args)
        if result_total is None:
            result_total = result.to_pandas()
        else:
            result_total = result_total.append(result.to_pandas(), ignore_index=True)
    return result_total

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

@arctern_udf('binary', 'binary')
def snap_to_road(roads, gps_points, num_thread=8):
    import pyarrow as pa
    arr_roads = pa.array(roads, type='binary')
    arr_gps_points = pa.array(gps_points, type='binary')
    arr_roads = _to_arrow_array_list(arr_roads)
    arr_gps_points = _to_arrow_array_list(arr_gps_points)
    result = arctern_core_.snap_to_road(arr_roads, arr_gps_points, num_thread)
    return _to_pandas_series(result)
