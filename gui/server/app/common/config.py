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

#import traceback


class Config:
    """
    manage all configuration items by load config.ini
    """
    __config = configparser.ConfigParser()

    def __init__(self):
        path = os.path.split(os.path.realpath(__file__))[
            0] + '/../../config.ini'
        self.__config.read(path)

    def get(self, section, key):
        """
        get value of config item
        """
        return self.__config.get(section, key)


INSTANCE = Config()
# traceback.print_stack()
