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

from abc import ABCMeta, abstractmethod

class DB(metaclass=ABCMeta):
    def dbtype(self):
        return self._db_type

    def id(self):
        return self._db_id

    def name(self):
        return self._db_name

    @abstractmethod
    def table_list(self):
        return self._table_list

    @abstractmethod
    def run(self, sql):
        pass

    @abstractmethod
    def load(self, metas):
        pass

    @abstractmethod
    def run_for_json(self, sql):
        pass

    @abstractmethod
    def get_table_info(self, table_name):
        pass

CENTER = {}
