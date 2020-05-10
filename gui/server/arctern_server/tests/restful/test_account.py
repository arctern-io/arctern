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

import requests

def test_login(host, port):
    url = 'http://' + host + ':' + port + '/login'
    # case 1: invalid username or password
    invalid_params = {
        'username': 'invaliduser',
        'password': 'invalidpassword'
    }
    response = requests.post(
        url=url,
        json=invalid_params
    )
    assert response.json()['code'] == -1
    assert response.json()['message'] == 'username/password error'
    assert response.json()['status'] == 'error'

    # case 2: correct username and password
    correct_params = {
        'username': 'zilliz',
        'password': '123456'
    }
    response = requests.post(
        url=url,
        json=correct_params
    )
    assert response.status_code == 200
