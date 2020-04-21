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

def pytest_addoption(parser):
    parser.addoption(
        '--host', action='store', default='127.0.0.1', help='the ip address of web server'
    )
    parser.addoption(
        '--port', action='store', default='8080', help='the port of web sever'
    )
    parser.addoption(
        '--config', action='store', default='../../db.json', help='the db config to be loaded'
    )

@pytest.fixture(scope='session')
def host(request):
    return request.config.getoption('--host')

@pytest.fixture(scope='session')
def port(request):
    return request.config.getoption('--port')

@pytest.fixture(scope='session')
def db_config(request):
    return request.config.getoption('--config')

@pytest.fixture(scope='session')
def token(host, port):
    url = 'http://' + host + ':' + port + '/login'
    response = requests.post(
        url=url,
        json={'username':'zilliz', 'password':'123456'}
    )
    return response.json()['data']['token']

@pytest.fixture(scope='session')
def headers(token):
    auth_header = {}
    auth_header['Authorization'] = 'Token ' + str(token)
    return auth_header
