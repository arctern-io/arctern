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

def test_dbs(headers):
    response = requests.get(
         url='http://192.168.2.29:8080/dbs',
         headers=headers
    )
    assert response.status_code == 200
    assert response.json()['data'][0]['id'] == '1'
    assert response.json()['data'][0]['name'] == 'nyc taxi'
    assert response.json()['data'][0]['type'] == 'spark'
