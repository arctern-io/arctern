# pylint: disable=redefined-outer-name

import pytest
from flask import Flask
from flask_cors import CORS

from app import service as app_service
from app.nyctaxi import data as nyctaxi_data

@pytest.fixture
def app():
    APP = Flask(__name__)
    APP.config['TESTING'] = True
    APP.register_blueprint(app_service.API)
    CORS(APP, resources=r'/*')
    nyctaxi_data.init()
    return APP

@pytest.fixture
def client(app):
    return app.test_client()
