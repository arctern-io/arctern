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

import requests
from retrying import retry

@retry(stop_max_attempt_number=3, wait_fixed=3000)
def safe_requests(*args, **kwargs):
    return requests.request(*args, **kwargs)
