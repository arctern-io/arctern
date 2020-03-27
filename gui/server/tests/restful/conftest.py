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

# pylint: disable=redefined-outer-name

import pytest
import requests

@pytest.fixture
def token():
    response = requests.post(
        url='http://192.168.2.29:8080/login',
        json={'username':'zilliz', 'password':'123456'}
    )
    return response.json()['data']['token']

@pytest.fixture
def headers(token):
    auth_header = {}
    auth_header['Authorization'] = 'Token ' + str(token)
    return auth_header
