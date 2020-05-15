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

# pylint: disable=logging-format-interpolation

import ast
import json

import requests

from flask import Blueprint, jsonify, request

from arctern_server.app.common import log
from arctern_server.app.common import config as app_config
from arctern_server.app import codegen
from arctern_server.app import pythonbackend

API = Blueprint("function_api", __name__)

def _function_forward(code, notebook_id, paragraph_id):
    # step 1: update paragraph
    if paragraph_id is None:
        url = app_config.ZEPPELEN_PREFIX + "/api/notebook/" + notebook_id + "/paragraph"
        payload = {
            "text": code,
        }   # TODO: more details
        log.INSTANCE.info("forward to: post {}\nbody: {}\n".format(url, payload))
        r = requests.post(url=url, json=payload)
        paragraph_id = r.json()["body"]
    else:
        url = app_config.ZEPPELEN_PREFIX + "/api/notebook/" + notebook_id + "/paragraph/" + paragraph_id
        payload = {
            "text": code,
        }   # TODO: more details
        log.INSTANCE.info("forward to: put {}".format(url))
        requests.put(url=url, json=payload)

    # step 2: run paragraph
    url = app_config.ZEPPELEN_PREFIX + "/api/notebook/run/" + notebook_id + "/" + paragraph_id
    log.INSTANCE.info("forward to: post {}".format(url))
    response = requests.post(url=url)
    return response.json()

@API.route('/v3/loadfile', methods=['POST'])
def load_file_zeppelin_interface():
    log.INSTANCE.info("POST /v3/loadfile: {}".format(request.json))

    interpreter_type = request.json.get("interpreter_type") or "pyspark"
    interpreter_name = request.json.get("interpreter_name") or "%spark.pyspark"
    notebook_id = request.json.get("notebook")
    paragraph_id = request.json.get("paragraph")

    if notebook_id is None:
        return jsonify(status="error", code=-1, message="no notebook specific!")

    load_code = interpreter + "\n\n"
    tables = request.json.get("tables")
    for table in tables:
        if interpreter_type == "pyspark":
            load_code += codegen.generate_load_code(table)
        elif interpreter_type == "python":
            load_code += pythonbackend.generate_load_code(table)
        else:
            raise Exception("Unsupported interpreter type!")
    log.INSTANCE.info("load code: {}".format(load_code))

    result = _function_forward(load_code, notebook_id, paragraph_id)
    status = result.get("status")
    if status != "OK":
        return jsonify(**result)
    return jsonify(status='success', code=200, message='load table successfully!')

@API.route('/v3/savefile', methods=['POST'])
def save_table_zeppelin_interface():
    log.INSTANCE.info("POST /v3/savefile: {}".format(request.json))

    interpreter_type = request.json.get("interpreter_type") or "pyspark"
    interpreter_name = request.json.get("interpreter_name") or "%spark.pyspark"
    notebook_id = request.json.get("notebook")
    paragraph_id = request.json.get("paragraph")

    if notebook_id is None:
        return jsonify(status="error", code=-1, message="no notebook specific!")

    save_code = interpreter + "\n\n"
    tables = request.json.get("tables")
    for table in tables:
        if interpreter_type == "pyspark":
            save_code += codegen.generate_save_code(table)
        elif interpreter_type == "python":
            save_code += pythonbackend.generate_save_code(table)
        else:
            raise Exception("Unsupported interpreter type!")
    log.INSTANCE.info("save code: {}".format(save_code))

    result = _function_forward(save_code, notebook_id, paragraph_id)
    status = result.get("status")
    if status != "OK":
        return jsonify(**result)
    return jsonify(status='success', code=200, message='save table successfully!')

# pylint: disable = too-many-branches
# pylint: disable = too-many-statements
@API.route('/v3/table/schema', methods=['GET'])
def table_schema_zeppelin_interface():
    log.INSTANCE.info("GET /v3/table/schema: {}".format(request.args))

    interpreter_type = request.args.get("interpreter_type") or "pyspark"
    interpreter_name = request.json.get("interpreter_name") or "%spark.pyspark"
    notebook_id = request.args.get("notebook")
    paragraph_id = request.args.get("paragraph")

    if notebook_id is None:
        return jsonify(status="error", code=-1, message="no notebook specific!")

    table_name = request.args.get("table")
    table_schema_code = interpreter + "\n\n"
    if interpreter_type == "pyspark":
        table_schema_code += codegen.generate_table_schema_code(table_name)
    elif interpreter_type == "python":
        table_schema_code += pythonbackend.generate_table_schema_code(table_name)
    else:
        raise Exception("Unsupported interpreter type!")
    log.INSTANCE.info("table_schema code: {}".format(table_schema_code))

    result = _function_forward(table_schema_code, notebook_id, paragraph_id)
    status = result.get("status")
    if status != "OK":
        return jsonify(**result)
    msgs = result["body"]["msg"]
    if interpreter_type == "pyspark":
        for msg in msgs:
            try:
                # convert list string to list
                data = ast.literal_eval(msg["data"])
                # convert json string to dict
                schema = [json.loads(item) for item in data]
            except Exception:   # pylint: disable=broad-except
                continue
            else:
                break
    elif interpreter_type == "python":
        for msg in msgs:
            try:
                data = msg["data"]
                split_line = data.split("\n")
                split_line = split_line[ : len(split_line) - 2]

                def _parse_line(line):
                    column_name = str()
                    column_type = str()
                    parse_state = 0
                    c_num = len(line)
                    for i in range(c_num):
                        if line[i] != ' ' and parse_state == 0:
                            column_name += line[i]
                        elif line[i] == ' ' and parse_state == 0:
                            parse_state = 1
                        elif parse_state == 1 and line[i] == ' ':
                            continue
                        elif parse_state == 1 and line[i] != ' ':
                            column_type += line[i]
                            parse_state = 2
                        elif parse_state == 2:
                            column_type += line[i]
                    return {column_name: column_type}

                schema = [_parse_line(line) for line in split_line]
            except Exception:   # pylint: disable=broad-except
                continue
            else:
                break
    return jsonify(
        status="success",
        code=200,
        table=table_name,
        schema=schema,
    )

@API.route('/v3/query', methods=['POST'])
def query_zeppelin_interface():
    log.INSTANCE.info("POST /v3/query: {}".format(request.json))

    interpreter_type = request.json.get("interpreter_type") or "pyspark"
    interpreter = request.json.get("interpreter") or "%spark.pyspark"
    notebook_id = request.json.get("notebook")
    paragraph_id = request.json.get("paragraph")

    if notebook_id is None:
        return jsonify(status="error", code=-1, message="no notebook specific!")

    query_code = interpreter + "\n\n"
    sql = request.json.get("sql")

    collect_result = request.json.get("collect_result")
    if interpreter_type == "pyspark":
        if collect_result is None or collect_result == "1":
            query_code += codegen.generate_run_for_json_code(sql)
        else:
            query_code += codegen.generate_run_sql_code(sql)
    else:
        raise Exception("Unsupported interpreter type!")

    log.INSTANCE.info("query code: {}".format(query_code))

    result = _function_forward(query_code, notebook_id, paragraph_id)
    status = result.get("status")
    if status != "OK":
        return jsonify(**result)
    if collect_result is None or collect_result == "1":
        msgs = result["body"]["msg"]
        for msg in msgs:
            try:
                data = ast.literal_eval(msg["data"])
            except Exception:   # pylint: disable=broad-except
                continue
            else:
                break
        return jsonify(
            status='success',
            code=200,
            result=[json.loads(item) for item in data],
            message="execute sql successfully!",
        )
    return jsonify(
        status='success',
        code=200,
        message="execute sql successfully!",
    )

def render_zeppelin_interface(payload, render_type):
    log.INSTANCE.info("POST /v3/{}: {}".format(render_type, payload))

    interpreter_type = request.json.get("interpreter_type") or "pyspark"
    interpreter = payload.get("interpreter") or "%spark.pyspark"
    notebook_id = payload.get("notebook")
    paragraph_id = payload.get("paragraph")

    if notebook_id is None:
        return jsonify(status="error", code=-1, message="no notebook specific!")

    render_code = interpreter + "\n\n"
    sql = payload.get("sql")
    params = payload.get("params")

    if interpreter_type == "pyspark":
        generate_func = {
            "pointmap": codegen.generate_pointmap_code,
            "heatmap": codegen.generate_heatmap_code,
            "choroplethmap": codegen.generate_choropleth_map_code,
            "weighted_pointmap": codegen.generate_weighted_map_code,
            "icon_viz": codegen.generate_icon_viz_code,
            "fishnetmap": codegen.generate_fishnetmap_code,
        }
        sql_code, vega_code = generate_func[render_type](sql, params)

        split_code = sql_code.split("\n")
        line_count = len(split_code)
        for i in range(line_count - 1):
            line = split_code[i]
            render_code += line + "\n"
        render_code += "res = {}\n".format(split_code[line_count - 1] + "\n")
        render_code += "vega = {}\n".format(vega_code)
        render_code += "data = {}(vega, res)\n".format(render_type)
    elif interpreter_type == "python":
        generate_func = {
            "pointmap": pythonbackend.generate_pointmap_code,
            "heatmap": pythonbackend.generate_heatmap_code,
            "choroplethmap": pythonbackend.generate_choropleth_map_code,
            "weighted_pointmap": pythonbackend.generate_weighted_map_code,
            "icon_viz": pythonbackend.generate_icon_viz_code,
            "fishnetmap": pythonbackend.generate_fishnetmap_code,
        }
        layer_func = {
            "pointmap": "point_map_layer",
            "heatmap": "heat_map_layer",
            "choroplethmap": "choropleth_map_layer",
            "weighted_pointmap": "weighted_point_map_layer",
            "icon_viz": "icon_viz_layer",
            "fishnetmap": "fishnet_map_layer",
        }
        import_code, input_code, vega_code = generate_func[render_type](sql, params)
        render_code += import_code
        render_code += "res = ({},)\n".format(input_code)
        render_code += "vega = {}\n".format(vega_code)
        if render_type == "weighted_pointmap":
            color_weights = request.json.get("color_weights")
            size_weights = request.json.get("size_weights")
            if color_weights or size_weights:
                render_code += "kwargs = dict()\n"
            if color_weights:
                render_code += "kwargs['color_weights'] = {}\n".format(color_weights)
            if size_weights:
                render_code += "kwargs['size_weights'] = {}\n".format(size_weights)
            if color_weights or size_weights:
                render_code += "data = {}(vega, *res, **kwargs)\n".format(layer_func[render_type])
            else:
                render_code += "data = {}(vega, *res)\n".format(layer_func[render_type])
        else:
            render_code += "data = {}(vega, *res)\n".format(layer_func[render_type])
    else:
        raise Exception("Unsupported interpreter type!")

    render_code += "imgStr = \"data:image/png;base64,\"\n"
    if interpreter_type == "pyspark":
        render_code += "imgStr += data\n"
    elif interpreter_type == "python":
        render_code += "imgStr += data.decode('utf-8')\n"
    render_code += "print( \"%html <img src='\" + imgStr + \"'>\" )\n"

    log.INSTANCE.info("{} code: {}".format(render_type, render_code))

    result = _function_forward(render_code, notebook_id, paragraph_id)
    status = result.get("status")
    if status != "OK":
        # return jsonify(**result)
        return result
    msgs = result["body"]["msg"]
    for msg in msgs:
        if msg["type"] == "HTML":
            data = msg["data"]
            break
    index = data.find(",")
    data = data[index + 1 : len(data) - 2]
    return {
        "status": 'success',
        "code": 200,
        "result": data,
    }

@API.route("/v3/pointmap", methods=['POST'])
def pointmap_zeppelin_interface():
    response = render_zeppelin_interface(request.json, "pointmap")
    return jsonify(**response)

@API.route("/v3/heatmap", methods=['POST'])
def heatmap_zeppelin_interface():
    response = render_zeppelin_interface(request.json, "heatmap")
    return jsonify(**response)

@API.route("/v3/choroplethmap", methods=['POST'])
def choroplethmap_zeppelin_interface():
    response = render_zeppelin_interface(request.json, "choroplethmap")
    return jsonify(**response)

@API.route("/v3/weighted_pointmap", methods=['POST'])
def weighted_pointmap_zeppelin_interface():
    response = render_zeppelin_interface(request.json, "weighted_pointmap")
    return jsonify(**response)

@API.route("/v3/icon_viz", methods=['POST'])
def icon_viz_zeppelin_interface():
    response = render_zeppelin_interface(request.json, "icon_viz")
    return jsonify(**response)

@API.route("/v3/fishnetmap", methods=['POST'])
def fishnetmap_zeppelin_interface():
    response = render_zeppelin_interface(request.json, "fishnetmap")
    return jsonify(**response)

@API.route("/v3/command", methods=['POST'])
def custom_command_zeppelin_interface():
    log.INSTANCE.info("POST /v3/command: {}".format(request.json))

    interpreter = request.json.get("interpreter") or "%spark.pyspark"
    notebook_id = request.json.get("notebook")
    paragraph_id = request.json.get("paragraph")

    if notebook_id is None:
        return jsonify(status="error", code=-1, message="no notebook specific!")

    command = interpreter + "\n\n"
    command += request.json.get("command")

    log.INSTANCE.info("command code: {}".format(command))

    result = _function_forward(command, notebook_id, paragraph_id)
    status = result.get("status")
    if status != "OK":
        return jsonify(**result)

    return jsonify(status="success", code=200, message="execute command successfully!")
