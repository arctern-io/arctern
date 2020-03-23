"""
Copyright (C) 2019-2020 Zilliz. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS S" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import getopt
import sys

from flask import Flask
from flask_cors import CORS

from app import service as app_service
from app.nyctaxi import data as nyctaxi_data
from app.common import config

APP = Flask(__name__)

APP.register_blueprint(app_service.API)

CORS(APP, resources=r'/*')


def usage():
    """
        help function
    """
    print('usage: python manange.py [options]')
    print('default: develop mode')
    print('-h: usage')
    print('-r: production mode')
    print('-i: ip address')
    print('-p: http port')


if __name__ == '__main__':
    IS_DEBUG = True
    IP = "0.0.0.0"
    PORT = config.INSTANCE.get("http", "port")

    try:
        OPTS, ARGS = getopt.getopt(sys.argv[1:], 'hri:p:')
    except getopt.GetoptError as _e:
        print("Error '{}' occured. Arguments {}.".format(str(_e), _e.args))
        usage()
        sys.exit(2)

    for opt, arg in OPTS:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt == '-r':
            IS_DEBUG = False
        elif opt == '-i':
            IP = arg
        elif opt == "-p":
            PORT = arg

    nyctaxi_data.init()

    if not IS_DEBUG:
        from waitress import serve
        serve(APP, host=IP, port=PORT)
    else:
        APP.debug = True
        APP.run(host=IP, port=PORT)
