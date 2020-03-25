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

import json

from app.common import spark

class TestSpark:

    def test_run(self):
        sample_num = 10
        sample_data = [(i, ) for i in range(sample_num)]

        session = spark.INSTANCE.session
        sample_df = session.createDataFrame(data=sample_data, schema=['sample'])
        sample_df.createGlobalTempView('test_run')

        res = spark.Spark.run("select * from global_temp.test_run").collect()
        for i in range(sample_num):
            assert res[i][0] == i

    def test_run_for_json(self):
        sample_num = 10
        sample_data = [(i, ) for i in range(sample_num)]

        session = spark.INSTANCE.session
        sample_df = session.createDataFrame(data=sample_data, schema=['sample'])
        sample_df.createGlobalTempView('test_run_for_json')

        res = spark.Spark.run_for_json("select * from global_temp.test_run_for_json")
        for i in range(sample_num):
            json_string = res[i]
            json_dict = json.loads(json_string)
            assert isinstance(json_dict, dict)
            assert len(json_dict) == 1
            for key, value in json_dict.items():
                assert key == 'sample'
                assert value == i
