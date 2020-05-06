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

# pylint: disable=logging-format-interpolation

import logging
import getopt
import sys
from pathlib import Path
import json

from flask import Flask, jsonify
from flask_cors import CORS

from arctern_server.app import service as app_service
from arctern_server.app import scope as app_scope
from arctern_server.app.common import log

APP = Flask(__name__)

APP.register_blueprint(app_service.API)
APP.register_blueprint(app_scope.API)

CORS(APP, resources=r'/*')

@APP.errorhandler(Exception)
def exception_handler(e):
    log.INSTANCE.error('exception: {}'.format(str(e)))
    return jsonify(status='error', code=-1, message=str(e))

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
    print('-c: json config to be loaded')
    print('--logfile=: path/to/logfile, default: ./log.txt')
    print('--loglevel=: log level [debug/info/warn/error/fatal], default: info')


# pylint: disable=too-many-branches
# pylint: disable=redefined-outer-name
def main(IS_DEBUG=True, IP="0.0.0.0", PORT=8080, JSON_CONFIG=None, LOG_FILE="/tmp/arctern_server_log.txt", LOG_LEVEL=logging.INFO):
    log.set_file(LOG_FILE, LOG_LEVEL)

    if JSON_CONFIG:
        json_file = Path(JSON_CONFIG)
        if not json_file.is_file():
            print("error: config %s doesn't exist!" % (JSON_CONFIG))
            sys.exit(0)
        else:
            with open(JSON_CONFIG, 'r') as f:
                content = json.load(f)
                _, code, message = app_service.load_data(content)
                print(message)
                if code != 200:
                    sys.exit(0)

    if not IS_DEBUG:
        from waitress import serve
        serve(APP, host=IP, port=PORT)
    else:
        APP.debug = True
        APP.run(host=IP, port=PORT)

if __name__ == '__main__':
    IS_DEBUG = True
    IP = "0.0.0.0"
    PORT = 8080
    JSON_CONFIG = None
    LOG_FILE = "log.txt"
    LOG_LEVEL = logging.INFO

    _LEVEL_DICT_ = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warn': logging.WARN,
        'error': logging.ERROR,
        'fatal': logging.FATAL
    }

    try:
        OPTS, ARGS = getopt.getopt(sys.argv[1:], 'hri:p:c:', ['logfile=', 'loglevel='])
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
        elif opt == '-c':
            JSON_CONFIG = arg
        elif opt == '--logfile':
            LOG_FILE = arg
        elif opt == '--loglevel':
            LOG_LEVEL = _LEVEL_DICT_.get(arg, logging.DEBUG)

    main(IS_DEBUG, IP, PORT, JSON_CONFIG, LOG_FILE, LOG_LEVEL)
