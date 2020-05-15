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

API = Blueprint("notebook_api", __name__)

@API.route('/notebook', methods=['GET', 'POST'])
def notebook():
    response = forward.forward_to_zeppelin(request)
    return jsonify(**response)

@API.route('/notebook/job/<note_id>', methods=['GET', 'POST'])
def get_or_run_all_paragraph(note_id):
    log.INSTANCE.info("note_id: {}".format(note_id))
    response = forward.forward_to_zeppelin(request)
    return jsonify(**response)

@API.route('/notebook/<note_id>', methods=['GET', 'DELETE'])
def get_or_remove_notebook(note_id):
    log.INSTANCE.info("note_id: {}".format(note_id))
    response = forward.forward_to_zeppelin(request)
    return jsonify(**response)

@API.route('/notebook/<note_id>/paragraph', methods=['POST'])
def create_paragraph(note_id):
    log.INSTANCE.info("note_id: {}".format(note_id))
    response = forward.forward_to_zeppelin(request)
    return jsonify(**response)

@API.route('/notebook/<note_id>/paragraph/<paragraph_id>', methods=['GET', 'PUT', 'DELETE'])
def get_paragraph_info(note_id, paragraph_id):
    log.INSTANCE.info("note_id: {}, paragraph_id: {}".format(note_id, paragraph_id))
    response = forward.forward_to_zeppelin(request)
    return jsonify(**response)

@API.route('/notebook/job/<note_id>/<paragraph_id>', methods=['GET', 'POST', 'DELETE'])
def run_paragraph_asynchronously(note_id, paragraph_id):
    log.INSTANCE.info("note_id: {}, paragraph_id: {}".format(note_id, paragraph_id))
    response = forward.forward_to_zeppelin(request)
    return jsonify(**response)
    # todo: filter response when request.method == 'POST'

@API.route('/notebook/run/<note_id>/<paragraph_id>', methods=['POST'])
def run_paragraph_synchronously(note_id, paragraph_id):
    log.INSTANCE.info("note_id: {}, paragraph_id: {}".format(note_id, paragraph_id))
    response = forward.forward_to_zeppelin(request)
    return jsonify(**response)
    # todo: filter response when request.method == 'POST'
