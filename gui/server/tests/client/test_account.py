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

from app.common import token as app_token

def test_token():
    expired = 7*24*60*60
    token = app_token.create('zilliz', expired)
    assert app_token.verify(token)

    token = app_token.create('invalidtoken', expired)
    assert not app_token.verify(token)

def test_login(client):
    response = client.post(
        '/login',
        data=json.dumps(dict(username='zilliz', password='123456')),
        content_type='application/json'
    )
    assert response.status_code == 200

    response = client.post(
        '/login',
        data=json.dumps(dict(username='invaliduser', password='invalidpassword')),
        content_type='application/json'
    )
    assert response.json['code'] == -1
    assert response.json['message'] == 'username/password error'
    assert response.json['status'] == 'error'
