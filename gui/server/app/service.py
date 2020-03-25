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

from flask import Blueprint, jsonify, request

from app import account
from app.common import token, utils

API = Blueprint('app_api', __name__)


'''
request:
    {
        "username": "arctern",
        "password": "*******"
    }
response:
    {
        "status": "success",
        "code": 200,
        "data": {
            "token": "xxxxx",
            "expired": "7d"
        },
        "message": null
    }
'''
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
    # print("password={}, db={}".format(password, accountDB.getPassword(username)))
    if not is_exist or password != real_pwd:
        return jsonify(status='error', code=-1, message='username/password error')

    #
    expired = 7*24*60*60

    content = {}
    content['token'] = token.create(request.json['username'], expired)
    content['expired'] = expired

    return jsonify(status='success', code=200, data=content)
