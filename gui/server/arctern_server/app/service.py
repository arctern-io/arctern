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

import json
from flask import Blueprint, jsonify, request

from arctern.util.vega import vega_choroplethmap, vega_heatmap, vega_pointmap, vega_weighted_pointmap, vega_icon, vega_fishnetmap
from arctern_pyspark import choroplethmap, heatmap, pointmap, weighted_pointmap, icon_viz, fishnetmap

from arctern_server.app import account
from arctern_server.app.common import spark, token, utils, db, log

API = Blueprint('app_api', __name__)

def load_data(content):
    if not utils.check_json(content, 'db_name') \
        or not utils.check_json(content, 'type'):
        return ('error', -1, 'no db_name or type field!')

    db_name = content['db_name']
    db_type = content['type']
    table_meta = content['tables']

    for _, db_instance in db.CENTER.items():
        if db_name == db_instance.name():
            db_instance.load(table_meta)
            return ('success', 200, 'load data succeed!')

    if db_type == 'spark':
        db_instance = spark.Spark(content)
        db_instance.load(table_meta)
        db.CENTER[db_instance.id()] = db_instance
        return ('success', 200, 'load data succeed!')

    return ('error', -1, 'sorry, but unsupported db type!')

@API.route('/load', methods=['POST'])
@token.AUTH.login_required
def load():
    """
    use this function to load data
    """
    log.INSTANCE.info('POST /load: {}'.format(request.json))
    status, code, message = load_data(request.json)
    return jsonify(status=status, code=code, message=message)

@API.route('/login', methods=['POST'])
def login():
    """
    login handler
    """
    log.INSTANCE.info('POST /login: {}'.format(request.json))

    if not utils.check_json(request.json, 'username') \
            or \
            not utils.check_json(request.json, 'password'):
        return jsonify(status='error', code=-1, message='username/password error')

    username = request.json['username']
    password = request.json['password']

    # verify username and pwd
    account_db = account.Account()
    is_exist, real_pwd = account_db.get_password(username)
    if not is_exist or (int)(password) != (int)(real_pwd):
        return jsonify(status='error', code=-1, message='username/password error')

    expired = 7*24*60*60

    content = {}
    content['token'] = token.create(request.json['username'], expired)
    content['expired'] = expired

    log.INSTANCE.info('/login: user: {}, toke: {}'.format(username, content['token']))

    return jsonify(status='success', code=200, data=content)


@API.route('/dbs')
@token.AUTH.login_required
def dbs():
    """
    /dbs handler
    """
    log.INSTANCE.info('GET /dbs:')

    content = []

    for _, db_instance in db.CENTER.items():
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
    log.INSTANCE.info('POST /db/tables: {}'.format(request.json))

    if not utils.check_json(request.json, 'id'):
        return jsonify(status='error', code=-1, message='json error: id is not exist')

    db_instance = db.CENTER.get(str(request.json['id']), None)
    if db_instance:
        content = db_instance.table_list()
        return jsonify(status="success", code=200, data=content)

    return jsonify(status="error", code=-1, message='there is no database whose id equal to ' + str(request.json['id']))


@API.route("/db/table/info", methods=['POST'])
@token.AUTH.login_required
def db_table_info():
    """
    /db/table/info handler
    """
    log.INSTANCE.info('POST /db/table/info: {}'.format(request.json))

    if not utils.check_json(request.json, 'id') \
            or not utils.check_json(request.json, 'table'):
        return jsonify(status='error', code=-1, message='query format error')

    content = []

    db_instance = db.CENTER.get(str(request.json['id']), None)
    if db_instance:
        if request.json['table'] not in db_instance.table_list():
            return jsonify(status="error", code=-1, message='the table {} is not in this db!'.format(request.json['table']))
        result = db_instance.get_table_info(request.json['table'])

        for row in result:
            obj = json.loads(row)
            content.append(obj)
        return jsonify(status="success", code=200, data=content)

    return jsonify(status="error", code=-1, message='there is no database whose id equal to ' + str(request.json['id']))

# pylint: disable=too-many-branches
@API.route("/db/query", methods=['POST'])
@token.AUTH.login_required
def db_query():
    """
    /db/query handler
    """
    log.INSTANCE.info('POST /db/query: {}'.format(request.json))

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

    db_instance = db.CENTER.get(str(request.json['id']), None)
    if db_instance is None:
        return jsonify(status="error", code=-1, message='there is no database whose id equal to ' + str(request.json['id']))

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
                int(query_params['point']['point_size']),
                query_params['point']['point_color'],
                float(query_params['point']['opacity']),
                query_params['point']['coordinate_system'])
            data = pointmap(vega, res)
            content['result'] = data
        elif query_type == 'heat':
            vega = vega_heatmap(
                int(query_params['width']),
                int(query_params['height']),
                query_params['heat']['bounding_box'],
                float(query_params['heat']['map_zoom_level']),
                query_params['heat']['coordinate_system'],
                query_params['heat']['aggregation_type'])
            data = heatmap(vega, res)
            content['result'] = data
        elif query_type == 'choropleth':
            vega = vega_choroplethmap(
                int(query_params['width']),
                int(query_params['height']),
                query_params['choropleth']['bounding_box'],
                query_params['choropleth']['color_gradient'],
                query_params['choropleth']['color_bound'],
                float(query_params['choropleth']['opacity']),
                query_params['choropleth']['coordinate_system'],
                query_params['choropleth']['aggregation_type'])
            data = choroplethmap(vega, res)
            content['result'] = data
        elif query_type == 'weighted':
            vega = vega_weighted_pointmap(
                int(query_params['width']),
                int(query_params['height']),
                query_params['weighted']['bounding_box'],
                query_params['weighted']['color_gradient'],
                query_params['weighted']['color_bound'],
                query_params['weighted']['size_bound'],
                float(query_params['weighted']['opacity']),
                query_params['weighted']['coordinate_system']
            )
            data = weighted_pointmap(vega, res)
            content['result'] = data
        elif query_type == 'icon':
            vega = vega_icon(
                int(query_params['width']),
                int(query_params['height']),
                query_params['icon']['bounding_box'],
                query_params['icon']['icon_path'],
                query_params['icon']['coordinate_system']
            )
            data = icon_viz(vega, res)
            content['result'] = data
        elif query_type == 'fishnet':
            vega = vega_fishnetmap(
                int(query_params['width']),
                int(query_params['height']),
                query_params['fishnet']['bounding_box'],
                query_params['fishnet']['color_gradient'],
                int(query_params['fishnet']['cell_size']),
                int(query_params['fishnet']['cell_spacing']),
                float(query_params['fishnet']['opacity']),
                query_params['fishnet']['coordinate_system'],
                query_params['fishnet']['aggregation_type']
            )
            data = fishnetmap(vega, res)
            content['result'] = data
        else:
            return jsonify(status="error",
                           code=-1,
                           message='{} not support'.format(query_type))

    return jsonify(status="success", code=200, data=content)
