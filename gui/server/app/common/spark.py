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

from pyspark.sql import SparkSession

from app.common import db
from arctern_pyspark import register_funcs


class Spark(db.DB):
    def __init__(self, db_config):
        envs = db_config['spark'].get('envs', None)
        if envs:    # for spark on yarn
            self._setup_driver_envs(envs)

        import uuid
        self._db_id = uuid.uuid1().int
        self._db_name = db_config['db_name']
        self._db_type = 'spark'
        self._table_list = []

        print("init spark begin")
        self.session = SparkSession.builder \
            .appName(db_config['spark']['app_name']) \
            .master(db_config['spark']['master-addr']) \
            .config("spark.executorEnv.PYSPARK_PYTHON", db_config['spark']['executor-python']) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.databricks.session.share", "false") \
            .getOrCreate()
        print("init spark done")
        register_funcs(self.session)

    def table_list(self):
        return self._table_list

    def _setup_driver_envs(self, envs):
        import os

        keys = ('PYSPARK_PYTHON', 'PYSPARK_DRIVER_PYTHON', 'JAVA_HOME',
                'HADOOP_CONF_DIR', 'YARN_CONF_DIR'
                )

        for key in keys:
            value = envs.get(key, None)
            if value:
                os.environ[key] = value

    def _create_session(self):
        """
        clone new session
        """
        return self.session.newSession()

    def run(self, sql):
        """
        submit sql to spark
        """
        session = self._create_session()
        register_funcs(session)
        return session.sql(sql)

    def run_for_json(self, sql):
        """
        convert the result of run() to json
        """
        _df = self.run(sql)
        return _df.coalesce(1).toJSON().collect()

    def load(self, metas):
        for meta in metas:
            if 'path' in meta and 'schema' in meta and 'format' in meta:
                options = meta.get('options', None)

                schema = str()
                for column in meta.get('schema'):
                    for key, value in column.items():
                        schema += key + ' ' + value + ', '
                rindex = schema.rfind(',')
                schema = schema[:rindex]

                df = self.session.read.format(meta.get('format')) \
                    .schema(schema) \
                    .load(meta.get('path'), **options)
                df.createOrReplaceGlobalTempView(meta.get('name'))
            elif 'sql' in meta:
                df = self.run(meta.get('sql', None))
                df.createOrReplaceGlobalTempView(meta.get('name'))

            if meta.get('visibility') == 'True':
                self._table_list.append('global_temp.' + meta.get('name'))

    def get_table_info(self, table_name):
        return self.run_for_json("desc table {}".format(table_name))
