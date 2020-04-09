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

from flask import Blueprint, jsonify, request

from app.common import token, utils
from app.common import error as app_error

API = Blueprint('app_api', __name__)

_DEFAULT_SCOPE = {}
_COMMAND_SCOPE = {
    'default': _DEFAULT_SCOPE,
}

@API.errorhandler(app_error.InvalidUsage)
def invalid_usage(ex):
    response = make_response(ex.message)
    response.status_code = ex.status_code
    return response

@API.route('/scope', methods=['GET', 'POST'])
@token.AUTH.login_required
def pscope():
    """
    GET: return all scope name
    POST: create new scope and return the name of new scope
    """
    if request.method == 'GET':
        return jsonify(status='success', code=200, data=_COMMAND_SCOPE.keys())
    elif request.method == 'POST':
        scope_id = request.json.get('scope_id', None)
        if scope_id is None:
            scope_id = str(uuid.uuid1()).replace('-', '')
            _COMMAND_SCOPE[scope_id] = dict()
            return jsonify(status='success', code=200, data={'scope_id': scope_id})
        elif scope_id is not in _COMMAND_SCOPE:
            _COMMAND_SCOPE[scope_id] = dict()
            return jsonify(status='success', code=200, data={'scope_id': scope_id})
        else:
            return jsonify(status='error', code=-1, message='sorry, scope_id exists!')

@API.route('/scope/<scope_id>', methods=['DELETE'])
@token.AUTH.login_required
def remove_scope():
    if scope_id not in _COMMAND_SCOPE or scope_id == 'default':
        return jsonify(status='error', code=-1, message='scope_id {} not found!'.format(scope_id))
    del _COMMAND_SCOPE[scope_id]
    return jsonify(status='success', code=200, message='remove scope {} successfully!'.format(scope_id))

@API.route('/command', methods=['POST'])
@token.AUTH.login_required
def pcommand():
    """
    execute python code with specific scope
    """
    scope_id = request.json.get('scope_id', default='default')
    code = request.json.get('command', None)
    if scope_id not in _COMMAND_SCOPE:
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
