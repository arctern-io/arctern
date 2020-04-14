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

from flask import Blueprint, jsonify, request, make_response

from app.common import token, utils
from app.common import error as app_error

API = Blueprint('command_api', __name__)

_COMMAND_SCOPE = {}

@API.errorhandler(app_error.InvalidUsage)
def invalid_usage(ex):
    response = make_response(ex.message)
    response.status_code = ex.status_code
    return response

@API.route('/scope', methods=['GET', 'POST'])
@token.AUTH.login_required
def create_scope():
    """
    GET: return all scope name
    POST: create new scope and return the name of new scope
    """
    if request.method == 'GET':
        return jsonify(status='success', code=200, data=list(_COMMAND_SCOPE.keys()))
    elif request.method == 'POST':
        scope_id = request.json.get('scope_id')
        if scope_id is None:
            scope_id = str(uuid.uuid1()).replace('-', '')
            _COMMAND_SCOPE[scope_id] = dict()
            return jsonify(status='success', code=200, data={'scope_id': scope_id})
        elif scope_id not in _COMMAND_SCOPE:
            _COMMAND_SCOPE[scope_id] = dict()
            return jsonify(status='success', code=200, data={'scope_id': scope_id})
        else:
            return jsonify(status='error', code=-1, message='sorry, scope_id exists!')

@API.route('/scope/<scope_id>', methods=['DELETE'])
@token.AUTH.login_required
def remove_scope(scope_id):
    print(scope_id)
    if scope_id not in _COMMAND_SCOPE:
        return jsonify(status='error', code=-1, message='scope_id {} not found!'.format(scope_id))
    del _COMMAND_SCOPE[scope_id]
    return jsonify(status='success', code=200, message='remove scope {} successfully!'.format(scope_id))

@API.route('/command', methods=['POST'])
@token.AUTH.login_required
def create_command():
    """
    execute python code with specific scope
    """
    scope_id = request.json.get('scope_id')
    code = request.json.get('command')
    if scope_id is None or scope_id not in _COMMAND_SCOPE:
        return jsonify(status='error', code=-1, message='scope_id {} not found!'.format(scope_id))
    if code is None:
        return jsonify(status='success', code=200, message='execute command successfully!')
    try:
        print("scope_id", scope_id)
        print(code)
        exec(code, _COMMAND_SCOPE[scope_id])
    except Exception as e:
        raise app_error.InvalidUsage(str(e), 400)
    else:
        return jsonify(status='success', code=200, message='execute command successfully!')

def _generate_env_code(envs):
    env_code = 'import os\n'
    keys = ('PYSPARK_PYTHON', 'PYSPARK_DRIVER_PYTHON', 'JAVA_HOME',
            'HADOOP_CONF_DIR', 'YARN_CONF_DIR', 'GDAL_DATA', 'PROJ_LIB'
            )
    if envs:
        for key in keys:
            value = envs.get(key, None)
            if value:
                env_code += 'os.environ["{}"] = "{}"\n'.format(key, value)
    return env_code

def _generate_session_code(metas):
    session_name = metas.get('session_name')
    uid = str(uuid.uuid1()).replace('-', '')
    app_name = metas.get('app_name', 'app_' + uid)
    master_addr = request.json.get('master-addr', 'local[*]')
    import socket
    localhost_ip = socket.gethostbyname(socket.gethostname())

    session_code = 'from pyspark.sql import SparkSession\n'
    session_code += '{} = SparkSession.builder'.format(session_name)
    session_code += '.appName("{}")'.format(app_name)
    session_code += '.master("{}")'.format(master_addr)
    session_code += '.config("spark.driver.host", "{}")'.format(localhost_ip)
    session_code += '.config("spark.sql.execution.arrow.pyspark.enabled", "true")'

    configs = metas.get('configs', None)
    if configs:
        for key, value in configs.items():
            if len(value) > 0:
                session_code += '.config("{}", "{}")'.format(key, value)
    session_code += '.getOrCreate()'
    return session_code

def _generate_load_code(session_name, table):
    table_name = table.get('name')
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
        load_code += '{}_df.createOrReplaceTempView("{}")'.format(table_name, table_name)
    elif 'sql' in table:
        sql = table.get('sql')
        load_code = '{}_df = {}.sql("{}")\n'.format(table_name, session_name, sql)
        load_code += '{}_df.createOrReplaceTempView("{}")'.format(table_name, table_name)
    return load_code

@API.route('/session', methods=['POST'])
@token.AUTH.login_required
def create_session():
    scope_id = request.json.get('scope_id')
    if scope_id is None or scope_id not in _COMMAND_SCOPE:
        return jsonify(status='error', code=-1, message='scope_id {} not found!'.format(scope_id))
    scope = _COMMAND_SCOPE[scope_id]
    session_name = request.json.get('session_name')
    if session_name is None:
        return jsonify(status='error', code=-1, message='no specific session name!')
    if session_name in scope.keys():
        return jsonify(status='error', code=-1, message='session name {} already in use!'.format(session_name))

    envs = request.json.get('envs')
    env_code = _generate_env_code(envs)
    print(env_code)

    session_code = _generate_session_code(request.json)
    print(session_code)

    try:
        exec(env_code, scope)
        exec(session_code, scope)
    except Exception as e:
        raise app_error.InvalidUsage(str(e), 400)
    else:
        return jsonify(status='success', code=200, message='create session successfully!')

@API.route('/session/<scope_id>/<session_name>', methods=['DELETE'])
@token.AUTH.login_required
def close_session(scope_id, session_name):
    if scope_id is None or scope_id not in _COMMAND_SCOPE:
        return jsonify(status='error', code=-1, message='scope_id {} not found!'.format(scope_id))
    scope = _COMMAND_SCOPE[scope_id]
    if session_name is None:
        return jsonify(status='error', code=-1, message='no specific session name!')
    if session_name not in scope.keys():
        return jsonify(status='error', code=-1, message='session name {} not found!'.format(session_name))
    session_code = '{}.stop()\n'.format(session_name)
    try:
        exec(session_code, scope)
    except Exception as e:
        raise app_error.InvalidUsage(str(e), 400)
    else:
        return jsonify(status='success', code=200, message='close session successfully!')

@API.route('/loadv2', methods=['POST'])
@token.AUTH.login_required
def load_table_v2():
    scope_id = request.json.get('scope_id')
    if scope_id is None or scope_id not in _COMMAND_SCOPE:
        return jsonify(status='error', code=-1, message='scope_id {} not found!'.format(scope_id))
    scope = _COMMAND_SCOPE[scope_id]
    session_name = request.json.get('session_name')
    if session_name is None:
        return jsonify(status='error', code=-1, message='no specific session name!')
    if session_name not in scope.keys():
        return jsonify(status='error', code=-1, message='session name {} not found!'.format(session_name))
    
    tables = request.json.get('tables')
    for table in tables:
        load_code = _generate_load_code(session_name, table)
        print(load_code)
        try:
            exec(load_code, scope)
        except Exception as e:
            raise app_error.InvalidUsage(str(e), 400)
    return jsonify(status='success', code=200, message='load table successfully!')