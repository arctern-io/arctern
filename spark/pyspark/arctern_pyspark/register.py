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

import inspect

def _get_funcs_in_module(module):
    for name in module.__all__:
        obj = getattr(module, name)
        if inspect.isfunction(obj):
            yield obj

def register_funcs(spark):
    from . import  _wrapper_func
    all_funcs = _get_funcs_in_module(_wrapper_func)
    for obj in all_funcs:
        #print(obj.__name__, obj)
        spark.udf.register(obj.__name__, obj)

def pointmap(df, vega):
    from . render_func import pointmap_2D
    res = pointmap_2D(df, vega)
    return res

def heatmap(df, vega):
    from . render_func import heatmap_2D
    res = heatmap_2D(df, vega)
    return res

def choroplethmap(df, vega):
    from . render_func import choroplethmap_2D
    res = choroplethmap_2D(df, vega)
    return res

def save_png(hex_data, file_name):
    from . render_func import save_png_2D
    save_png_2D(hex_data, file_name)