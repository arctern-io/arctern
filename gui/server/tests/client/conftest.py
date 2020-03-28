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
from flask import Flask
from flask_cors import CORS

from app.common import token as app_token
from app import service as app_service
from app.nyctaxi import data as nyctaxi_data

@pytest.fixture
def app():
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(app_service.API)
    CORS(app, resources=r'/*')
    nyctaxi_data.init()
    return app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def token():
    expired = 7*24*60*60
    return app_token.create("zilliz", expired)

@pytest.fixture
def headers(token):
    """
    use this header to make sure authorization passed
    """
    auth_header = {}
    auth_header['Authorization'] = 'Token ' + str(token)
    return auth_header
