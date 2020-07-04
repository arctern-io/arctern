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

# pylint: disable=logging-format-interpolation

from arctern_server.app.common import log
from arctern_server.app.common import config as app_config
from arctern_server.app import httpretry

def forward_to_zeppelin(request):
    url_path = request.path
    zeppelin_url = app_config.ZEPPELEN_PREFIX + "/api" + url_path
    method = request.method

    log.INSTANCE.info("{} {}, body: {}, args: {}".format(method, url_path, request.json, request.args))
    log.INSTANCE.info("forward to: {}".format(zeppelin_url))

    r = httpretry.safe_requests(method, zeppelin_url, data=request.data, headers=request.headers)
    return r.json()
