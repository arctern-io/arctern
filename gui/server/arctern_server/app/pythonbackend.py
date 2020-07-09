"""
Copyright (C) 2019-2020 Zilliz. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import uuid

# pylint: disable=too-many-branches
def generate_load_code(table):
    load_code = "import pandas\n"
    table_name = table.get("name")
    path = table.get("path")
    table_format = table.get("format")
    schema = dict()
    uid = str(uuid.uuid1()).replace("-", "")
    if "schema" in table:
        for column in table.get('schema'):
            for key, value in column.items():
                schema[key] = value
    if table_format == "csv":
        options = table.get("options")
        if "header" in options:
            header = options.get("header")
            options["header"] = 0 if header == "True" else None
        load_code += 'options_{} = {}\n'.format(uid, options)
        if schema:
            load_code += 'schema_{} = {}\n'.format(uid, schema)
            load_code += '{0} = pandas.read_csv("{1}", dtype=schema_{2}, **options_{2})'.format(table_name, path, uid)
        else:
            load_code += '{0} = pandas.read_csv("{1}", **options_{2})'.format(table_name, path, uid)
    elif table_format == "json":
        if schema:
            load_code += 'schema_{} = {}\n'.format(uid, schema)
            load_code += '{0} = pandas.read_json("{1}", dtype=schema_{2})'.format(table_name, path, uid)
        else:
            load_code += '{0} = pandas.read_json("{1}")'.format(table_name, path)
    elif table_format == "parquet":
        load_code += '{0} = pandas.read_parquet("{1}")'.format(table_name, path)
    elif table_format == "orc":
        load_code += '{0} = pandas.read_orc("{1}")'.format(table_name, path)
    else:
        raise Exception("Unsupported file format!")

    return load_code

def generate_save_code(table):
    path = table.get("path")
    file_format = table.get("format")
    options = table.get("options")
    if "header" in options:
        header = options.get("header")
        options["header"] = 0 if header == "True" else None
    table_name = table.get("table_name")

    if options:
        uid = str(uuid.uuid1()).replace("-", "")
        save_code = "options_{} = {}\n".format(uid, options)
        save_code += '{0}.to_{1}("{2}", **options_{3})'.format(table_name, file_format, path, uid)
    else:
        save_code = '{0}.to_{1}("{2}")'.format(table_name, file_format, path)
    return save_code

def generate_table_schema_code(table_name):
    schema_code = "import pandas\n"
    schema_code += "{0}.dtypes\n".format(table_name)
    return schema_code

def generate_pointmap_code(input_data, params):
    import_code = 'from arctern.util.vega import vega_pointmap\n'
    import_code += 'from arctern import *\n'
    import_code += 'import os, sys\n'
    import_code += 'os.environ["PROJ_LIB"] = sys.prefix + "/share/proj"\n'
    import_code += 'os.environ["GDAL_DATA"] = sys.prefix + "/share/gdal"\n'
    vega_code = 'vega_pointmap({}, {}, {}, {}, "{}", {}, "{}")'.format(
        int(params.get('width')),
        int(params.get('height')),
        params.get('bounding_box'),
        int(params.get('point_size')),
        params.get('point_color'),
        float(params.get('opacity')),
        params.get('coordinate_system')
    )
    points = input_data.get("points")
    return import_code, points, vega_code

def generate_weighted_map_code(input_data, params):
    import_code = 'from arctern.util.vega import vega_weighted_pointmap\n'
    import_code += 'from arctern import *\n'
    import_code += 'import os, sys\n'
    import_code += 'os.environ["PROJ_LIB"] = sys.prefix + "/share/proj"\n'
    import_code += 'os.environ["GDAL_DATA"] = sys.prefix + "/share/gdal"\n'
    vega_code = 'vega_weighted_pointmap({}, {}, {}, {}, {}, {}, {}, "{}")'.format(
        int(params.get('width')),
        int(params.get('height')),
        params.get('bounding_box'),
        params.get('color_gradient'),
        params.get('color_bound'),
        params.get('size_bound'),
        float(params.get('opacity')),
        params.get('coordinate_system')
    )
    params_code = ""
    params_code += input_data.get("points")
    color_weights = input_data.get("color_weights")
    size_weights = input_data.get("size_weights")
    if color_weights:
        params_code += ", color_weights={}".format(color_weights)
    if size_weights:
        params_code += ", size_weights={}".format(size_weights)
    return import_code, params_code, vega_code

def generate_heatmap_code(input_data, params):
    import_code = 'from arctern.util.vega import vega_heatmap\n'
    import_code += 'from arctern import *\n'
    import_code += 'import os, sys\n'
    import_code += 'os.environ["PROJ_LIB"] = sys.prefix + "/share/proj"\n'
    import_code += 'os.environ["GDAL_DATA"] = sys.prefix + "/share/gdal"\n'
    vega_code = 'vega_heatmap({}, {}, {}, {}, "{}", "{}")'.format(
        int(params.get('width')),
        int(params.get('height')),
        params.get('bounding_box'),
        float(params.get('map_zoom_level')),
        params.get('coordinate_system'),
        params.get('aggregation_type')
    )
    params_code = ""
    params_code += input_data.get("points")
    params_code += ", " + input_data.get("weights")
    return import_code, params_code, vega_code

def generate_choropleth_map_code(input_data, params):
    import_code = 'from arctern.util.vega import vega_choroplethmap\n'
    import_code += 'from arctern import *\n'
    import_code += 'import os, sys\n'
    import_code += 'os.environ["PROJ_LIB"] = sys.prefix + "/share/proj"\n'
    import_code += 'os.environ["GDAL_DATA"] = sys.prefix + "/share/gdal"\n'
    vega_code = 'vega_choroplethmap({}, {}, {}, {}, {}, {}, "{}", "{}")'.format(
        int(params.get('width')),
        int(params.get('height')),
        params.get('bounding_box'),
        params.get('color_gradient'),
        params.get('color_bound'),
        float(params.get('opacity')),
        params.get('coordinate_system'),
        params.get('aggregation_type')
    )
    params_code = ""
    params_code += input_data.get("region_boundaries")
    params_code += ", " + input_data.get("weights")
    return import_code, params_code, vega_code

def generate_icon_viz_code(input_data, params):
    import_code = 'from arctern.util.vega import vega_icon\n'
    import_code += 'from arctern import *\n'
    import_code += 'import os, sys\n'
    import_code += 'os.environ["PROJ_LIB"] = sys.prefix + "/share/proj"\n'
    import_code += 'os.environ["GDAL_DATA"] = sys.prefix + "/share/gdal"\n'
    vega_code = 'vega_icon({}, {}, {}, "{}", {}, "{}")'.format(
        int(params.get('width')),
        int(params.get('height')),
        params.get('bounding_box'),
        params.get('icon_path'),
        params.get('icon_size'),
        params.get('coordinate_system')
    )
    points = input_data.get("points")
    return import_code, points, vega_code

def generate_fishnetmap_code(input_data, params):
    import_code = 'from arctern.util.vega import vega_fishnetmap\n'
    import_code += 'from arctern import *\n'
    import_code += 'import os, sys\n'
    import_code += 'os.environ["PROJ_LIB"] = sys.prefix + "/share/proj"\n'
    import_code += 'os.environ["GDAL_DATA"] = sys.prefix + "/share/gdal"\n'
    vega_code = 'vega_fishnetmap({}, {}, {}, {}, {}, {}, {}, "{}", "{}")'.format(
        int(params.get('width')),
        int(params.get('height')),
        params.get('bounding_box'),
        params.get('color_gradient'),
        int(params.get('cell_size')),
        int(params.get('cell_spacing')),
        float(params.get('opacity')),
        params.get('coordinate_system'),
        params.get('aggregation_type')
    )
    params_code = ""
    params_code += input_data.get("points")
    params_code += ", " + input_data.get("weights")
    return import_code, params_code, vega_code
