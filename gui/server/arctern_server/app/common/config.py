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

import configparser
import os

class MyConf(configparser.ConfigParser):# pylint: disable=too-many-ancestors
    #preserve case for letters
    def optionxform(self, optionstr):
        return optionstr

INSTANCE = MyConf()
INSTANCE.read(os.path.split(os.path.realpath(__file__))[0]
              + '/../../config.ini')

ZEPPELEN_HOST = INSTANCE.get("zeppelin", "zeppelin-host", fallback="127.0.0.1")
ZEPPELEN_PORT = INSTANCE.get("zeppelin", "zeppelin-port", fallback=8888)
ZEPPELEN_PREFIX = "http://" + ZEPPELEN_HOST + ":" + ZEPPELEN_PORT
