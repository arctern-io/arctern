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

import requests

from flask import Blueprint, jsonify, request

from arctern_server.app.common import log
from arctern_server.app.common import config as app_config
from arctern_server.app import codegen

API = Blueprint("function_api", __name__)

def _function_forward(code, notebook_id, paragraph_id):
    # step 1: update paragraph
    url = app_config.ZEPPELEN_PREFIX + "/api/notebook/" + notebook_id + "/paragraph/" + paragraph_id
    payload = {
        "text": code,
    }   # TODO: more details
    log.INSTANCE.info("forward to: {}".format(url))
    requests.put(url=url, json=payload)

    # step 2: run paragraph
    url = app_config.ZEPPELEN_PREFIX + "/api/notebook/run/" + notebook_id + "/" + paragraph_id
    log.INSTANCE.info("forward to: {}".format(url))
    response = requests.post(url=url)
    return response.json()

@API.route('/v3/loadfile', methods=['POST'])
def load_file_zeppelin_interface():
    log.INSTANCE.info("POST /v3/loadfile: {}".format(request.json))

    interpreter = request.json.get("interpreter") or "%spark.pyspark"
    notebook_id = request.json.get("notebook")
    paragraph_id = request.json.get("paragraph")

    if notebook_id is None or paragraph_id is None:
        return jsonify(status="error", code=-1, message="no notebook or paragraph assigned!")

    load_code = interpreter + "\n\n"
    tables = request.json.get("tables")
    for table in tables:
        load_code += codegen.generate_load_code(table)
    log.INSTANCE.info("load code: {}".format(load_code))

    result = _function_forward(load_code, notebook_id, paragraph_id)
    return jsonify(**result)    # TODO: filter

@API.route('/v3/savetable', methods=['POST'])
def save_table_zeppelin_interface():
    log.INSTANCE.info("POST /v3/savetable: {}".format(request.json))

    interpreter = request.json.get("interpreter") or "%spark.pyspark"
    notebook_id = request.json.get("notebook")
    paragraph_id = request.json.get("paragraph")

    if notebook_id is None or paragraph_id is None:
        return jsonify(status="error", code=-1, message="no notebook or paragraph assigned!")

    save_code = interpreter + "\n\n"
    tables = request.json.get("tables")
    for table in tables:
        save_code += codegen.generate_save_code(table)
    log.INSTANCE.info("save code: {}".format(save_code))

    result = _function_forward(save_code, notebook_id, paragraph_id)
    return jsonify(**result)    # TODO: filter

@API.route('/v3/table/schema', methods=['GET'])
def table_schema_zeppelin_interface(notebook_id, paragraph_id):
    log.INSTANCE.info("GET /v3/table/schema: {}".format(requests.args))

    interpreter = request.args.get("interpreter") or "%spark.pyspark"
    notebook_id = request.args.get("notebook")
    paragraph_id = request.args.get("paragraph")

    if notebook_id is None or paragraph_id is None:
        return jsonify(status="error", code=-1, message="no notebook or paragraph assigned!")

    table_name = request.args.get("table")
    table_schema_code = interpreter + "\n\n"
    table_schema_code += codegen.generate_table_schema_code(table_name)
    log.INSTANCE.info("table_schema code: {}".format(table_schema_code))

    result = _function_forward(table_schema_code, notebook_id, paragraph_id)
    return jsonify(**result)    # TODO: filter

@API.route('/v3/query', methods=['POST'])
def query_zeppelin_interface():
    log.INSTANCE.info("POST /v3/query: {}".format(request.json))

    interpreter = request.args.get("interpreter") or "%spark.pyspark"
    notebook_id = request.args.get("notebook")
    paragraph_id = request.args.get("paragraph")

    if notebook_id is None or paragraph_id is None:
        return jsonify(status="error", code=-1, message="no notebook or paragraph assigned!")

    query_code = interpreter + "\n\n"
    sql = requests.json.get("sql")
    query_code += codegen.generate_run_sql_code(sql)
    log.INSTANCE.info("query code: {}".format(query_code))

    result = _function_forward(query_code, notebook_id, paragraph_id)
    return jsonify(**result)    # TODO: filter

@API.route('/v3/pointmap', methods=['POST'])
def pointmap_zeppelin_interface():
    log.INSTANCE.info("POST /v3/query: {}".format(request.json))

    interpreter = request.args.get("interpreter") or "%spark.pyspark"
    notebook_id = request.args.get("notebook")
    paragraph_id = request.args.get("paragraph")

    if notebook_id is None or paragraph_id is None:
        return jsonify(status="error", code=-1, message="no notebook or paragraph assigned!")

    render_code = interpreter + "\n\n"
    sql = request.json.get("sql")
    params = request.json.get("params")
    sql_code, vega_code = codegen.generate_pointmap_code(sql, params)

    render_code += "res = {}\n".format(sql_code)
    render_code += "vega = {}\n".format(vega_code)
    render_code += "data = pointmap(vega, res)\n"
    render_code += "imgStr = \"data:image/png;base64,\""
    render_code += "imgStr += data"
    render_code += "print( \"%html <img src='\" + imgStr + \"'>\" )"

    log.INSTANCE.info("query code: {}".format(render_code))

    result = _function_forward(render_code, notebook_id, paragraph_id)
    return jsonify(**result)    # TODO: filter
