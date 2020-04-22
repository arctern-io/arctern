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

import logging
from logging.handlers import RotatingFileHandler

INSTANCE = logging.getLogger()

def set_file(path, level=logging.DEBUG):
    INSTANCE.setLevel(level)
    handler = RotatingFileHandler(path, maxBytes=10*1024*1024, backupCount=30)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s[%(lineno)d] %(message)s')
    handler.setFormatter(formatter)
    INSTANCE.addHandler(handler)

if __name__ == '__main__':
    set_file("log.txt", logging.INFO)
    INSTANCE.debug("start test")
    INSTANCE.info("start test")
    INSTANCE.warning("start test")
    INSTANCE.error("start test")
    INSTANCE.fatal("start test")
