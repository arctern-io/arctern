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

import json
from arctern.util.vega import vega_choroplethmap, vega_heatmap, vega_pointmap
from arctern_pyspark import choroplethmap, heatmap, pointmap

from flask import Blueprint, jsonify, request

from app import account
from app.common import spark, token, utils

API = Blueprint('app_api', __name__)

DB_MAP = {}

@API.route('/load', methods=['POST'])
@token.AUTH.login_required
def load():
    """
    use this function to load data
    """
    if not utils.check_json(request.json, 'db_name') \
        or not utils.check_json(request.json, 'type'):
        return jsonify(status='error', code=-1, message='no db_name or type field!')

    db_name = request.json['db_name']
    db_type = request.json['type']

    if db_name in DB_MAP:
        db_instance = DB_MAP[db_name]
        table_meta = request.json['tables']
        db_instance.load(table_meta)
        return jsonify(status='success', code=200, message='load data succeed!')
    elif db_type == 'spark':
        DB_MAP[db_name] = spark.Spark(request.json)
        table_meta = request.json['tables']
        DB_MAP[db_name].load(table_meta)
        return jsonify(status='success', code=200, message='load data succeed!')
    else:
        return jsonify(status='error', code=-1, message='sorry, but unsupported db type!')

@API.route('/login', methods=['POST'])
def login():
    """
    login handler
    """
    if not utils.check_json(request.json, 'username') \
            or \
            not utils.check_json(request.json, 'password'):
        return jsonify(status='error', code=-1, message='username/password error')

    username = request.json['username']
    password = request.json['password']

    # verify username and pwd
    account_db = account.Account()
    is_exist, real_pwd = account_db.get_password(username)
    if not is_exist or password != real_pwd:
        return jsonify(status='error', code=-1, message='username/password error')

    expired = 7*24*60*60

    content = {}
    content['token'] = token.create(request.json['username'], expired)
    content['expired'] = expired

    return jsonify(status='success', code=200, data=content)


@API.route('/dbs')
@token.AUTH.login_required
def dbs():
    """
    /dbs handler
    """
    content = []

    for _, db_instance in DB_MAP.items():
        info = {}
        info['id'] = db_instance.id()
        info['name'] = db_instance.name()
        info['type'] = db_instance.dbtype()
        content.append(info)

    return jsonify(status='success', code=200, data=content)


@API.route("/db/tables", methods=['POST'])
@token.AUTH.login_required
def db_tables():
    """
    /db/tables handler
    """
    if not utils.check_json(request.json, 'id'):
        return jsonify(status='error', code=-1, message='json error: id is not exist')

    for _, instance in DB_MAP.items():
        if instance.id() == int(request.json['id']):
            content = instance.table_list()
            return jsonify(status="success", code=200, data=content)

    return jsonify(status="error", code=-1, message='there is no database whose id equal to ' + request.json['id'])


@API.route("/db/table/info", methods=['POST'])
@token.AUTH.login_required
def db_table_info():
    """
    /db/table/info handler
    """
    if not utils.check_json(request.json, 'id') \
            or not utils.check_json(request.json, 'table'):
        return jsonify(status='error', code=-1, message='query format error')

    content = []

    for _, instance in DB_MAP.items():
        if instance.id() == int(request.json['id']):
            if request.json['table'] not in instance.table_list():
                return jsonify(status="error", code=-1, message='the table {} is not in this db!'.format(request.json['table']))
            result = instance.get_table_info(request.json['table'])

        for row in result:
            obj = json.loads(row)
            content.append(obj)
        return jsonify(status="success", code=200, data=content)

    return jsonify(status="error", code=-1, message='there is no database whose id equal to ' + request.json['id'])


@API.route("/db/query", methods=['POST'])
@token.AUTH.login_required
def db_query():
    """
    /db/query handler
    """
    if not utils.check_json(request.json, 'id') \
            or not utils.check_json(request.json, 'query') \
            or not utils.check_json(request.json['query'], 'type') \
            or not utils.check_json(request.json['query'], 'sql'):
        return jsonify(status='error', code=-1, message='query format error')

    query_sql = request.json['query']['sql']
    query_type = request.json['query']['type']

    content = {}
    content['sql'] = query_sql
    content['err'] = False

    db_instance = None

    for _, instance in DB_MAP.items():
        if instance.id() == int(request.json['id']):
            db_instance = instance
    if db_instance is None:
        return jsonify(status="error", code=-1, message='there is no database whose id equal to ' + request.json['id'])

    if query_type == 'sql':
        res = db_instance.run_for_json(query_sql)
        data = []
        for row in res:
            obj = json.loads(row)
            data.append(obj)
        content['result'] = data
    else:
        if not utils.check_json(request.json['query'], 'params'):
            return jsonify(status='error', code=-1, message='query format error')
        query_params = request.json['query']['params']

        res = db_instance.run(query_sql)

        if query_type == 'point':
            vega = vega_pointmap(
                int(query_params['width']),
                int(query_params['height']),
                query_params['point']['bounding_box'],
                int(query_params['point']['stroke_width']),
                query_params['point']['stroke'],
                float(query_params['point']['opacity']),
                query_params['point']['coordinate'])
            data = pointmap(res, vega)
            content['result'] = data
        elif query_type == 'heat':
            vega = vega_heatmap(
                int(query_params['width']),
                int(query_params['height']),
                float(query_params['heat']['map_scale']),
                query_params['heat']['bounding_box'],
                query_params['heat']['coordinate'])
            data = heatmap(res, vega)
            content['result'] = data
        elif query_type == 'choropleth':
            vega = vega_choroplethmap(
                int(query_params['width']),
                int(query_params['height']),
                query_params['choropleth']['bounding_box'],
                query_params['choropleth']['color_style'],
                query_params['choropleth']['rule'],
                float(query_params['choropleth']['opacity']),
                query_params['choropleth']['coordinate'])
            data = choroplethmap(res, vega)
            content['result'] = data
        else:
            return jsonify(status="error",
                           code=-1,
                           message='{} not support'.format(query_type))
    return jsonify(status="success", code=200, data=content)
