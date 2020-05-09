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

def generate_session_code(session_name="spark"):
    uid = str(uuid.uuid1()).replace("-", "")
    app_name = "app_" + uid
    from arctern_server.app.common import config as app_config
    master_addr = app_config.INSTANCE.get("spark", "master-addr", fallback="local[*]")
    import socket
    localhost_ip = socket.gethostbyname(socket.gethostname())

    session_code = 'from arctern.util.vega import vega_choroplethmap, vega_heatmap, vega_pointmap, vega_weighted_pointmap, vega_icon, vega_fishnetmap\n'
    session_code += 'from arctern_pyspark import choroplethmap, heatmap, pointmap, weighted_pointmap, icon_viz, fishnetmap\n'
    session_code += 'from arctern_pyspark import register_funcs\n'
    session_code += 'from pyspark.sql import SparkSession\n'
    session_code += '{} = SparkSession.builder'.format(session_name)
    session_code += '.appName("{}")'.format(app_name)
    session_code += '.master("{}")'.format(master_addr)
    session_code += '.config("spark.sql.warehouse.dir", "/tmp")'
    session_code += '.config("spark.driver.host", "{}")'.format(localhost_ip)
    session_code += '.config("spark.sql.execution.arrow.pyspark.enabled", "true")'
    session_code += '.getOrCreate()\n'
    session_code += 'register_funcs({})\n'.format(session_name)

    return session_code

def generate_load_code(table, session_name="spark"):
    table_name = table.get("name")
    if 'path' in table and 'schema' in table and 'format' in table:
        options = table.get('options', None)

        schema = str()
        for column in table.get('schema'):
            for key, value in column.items():
                schema += key + ' ' + value + ', '
        rindex = schema.rfind(',')
        schema = schema[:rindex]

        table_format = table.get('format')
        path = table.get('path')
        load_code = '{}_df = {}.read'.format(table_name, session_name)
        load_code += '.format("{}")'.format(table_format)
        load_code += '.schema("{}")'.format(schema)
        for key, value in options.items():
            load_code += '.option("{}", "{}")'.format(key, value)
        load_code += '.load("{}")\n'.format(path)
        load_code += '{0}_df.createOrReplaceTempView("{0}")\n'.format(table_name)
    elif 'sql' in table:
        sql = table.get('sql')
        load_code = '{}_df = {}.sql("{}")\n'.format(table_name, session_name, sql)
        load_code += '{0}_df.createOrReplaceTempView("{0}")\n'.format(table_name)
    return load_code

def generate_save_code(table, session_name="spark"):
    path = table.get("path")
    table_format = table.get("format")
    options = table.get("options", None)
    sql = table.get("sql")

    save_code = '{}.sql("{}")'.format(session_name, sql)
    # no need to coalesce(1) just due to saving as a single file?
    save_code += '.coalesce(1).write.format("{}")'.format(table_format)
    for key, value in options.items():
        save_code += '.option("{}", "{}")'.format(key, value)
    save_code += '.save("{}")\n'.format(path)

    return save_code

def generate_run_sql_code(sql, session_name='spark'):
    code = '{}.sql("{}")'.format(session_name, sql)
    return code

def generate_run_for_json_code(sql, session_name='spark'):
    code = '{}.sql("{}")'.format(session_name, sql)
    code += '.coalesce(1).toJSON().collect()'
    return code

def generate_table_schema_code(table_name, session_name='spark'):
    sql = "desc table {}".format(table_name)
    return generate_run_for_json_code(sql, session_name)

def generate_table_count_code(table_name, session_name='spark'):
    sql = "select count(*) as num_rows from {}".format(table_name)
    return generate_run_for_json_code(sql, session_name)

def generate_pointmap_code(sql, params, session_name='spark'):
    sql_code = generate_run_sql_code(sql, session_name)
    vega_code = 'vega_pointmap({}, {}, {}, {}, "{}", {}, "{}")'.format(
        int(params.get('width')),
        int(params.get('height')),
        params.get('bounding_box'),
        int(params.get('point_size')),
        params.get('point_color'),
        float(params.get('opacity')),
        params.get('coordinate_system')
    )
    return sql_code, vega_code

def generate_heatmap_code(sql, params, session_name='spark'):
    sql_code = generate_run_sql_code(sql, session_name)
    vega_code = 'vega_heatmap({}, {}, {}, {}, "{}", "{}")'.format(
        int(params.get('width')),
        int(params.get('height')),
        params.get('bounding_box'),
        float(params.get('map_zoom_level')),
        params.get('coordinate_system'),
        params.get('aggregation_type')
    )
    return sql_code, vega_code

def generate_choropleth_map_code(sql, params, session_name='spark'):
    sql_code = generate_run_sql_code(sql, session_name)
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
    return sql_code, vega_code

def generate_weighted_map_code(sql, params, session_name='spark'):
    sql_code = generate_run_sql_code(sql, session_name)
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
    return sql_code, vega_code

def generate_icon_viz_code(sql, params, session_name='spark'):
    sql_code = generate_run_sql_code(sql, session_name)
    vega_code = 'vega_icon({}, {}, {}, "{}", "{}")'.format(
        int(params.get('width')),
        int(params.get('height')),
        params.get('bounding_box'),
        params.get('icon_path'),
        params.get('coordinate_system')
    )
    return sql_code, vega_code

def generate_fishnetmap_code(sql, params, session_name='spark'):
    sql_code = generate_run_sql_code(sql, session_name)
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
    return sql_code, vega_code
