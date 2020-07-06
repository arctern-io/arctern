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

from flask import Blueprint, jsonify, request

from arctern_server.app import forward
from arctern_server.app.common import log

API = Blueprint("interpreter_api", __name__)

@API.route('/interpreter', methods=['GET'])
def get_all_interpreter():
    response = forward.forward_to_zeppelin(request)
    return jsonify(**response)

@API.route('/interpreter/setting', methods=['GET', 'POST'])
def interpreter_setting():
    response = forward.forward_to_zeppelin(request)
    return jsonify(**response)

@API.route('/interpreter/setting/<setting_id>', methods=['GET', 'PUT', 'DELETE'])
def interpreter_setting_by_id(setting_id):
    log.INSTANCE.info("setting_id: {}".format(setting_id))
    response = forward.forward_to_zeppelin(request)
    return jsonify(**response)

@API.route('/interpreter/setting/restart/<setting_id>', methods=['PUT'])
def restart_interpreter(setting_id):
    log.INSTANCE.info("setting_id: {}".format(setting_id))
    response = forward.forward_to_zeppelin(request)
    return jsonify(**response)
