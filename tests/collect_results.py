# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import random
import os
# import inspect
import shutil
import glob
from util import arctern_result
from util import spark_result
from util import get_tests


def collect_results():
    """Collect spark results from different dirs."""
    if not os.path.isdir(arctern_result):
        os.makedirs(arctern_result)

    names, table_names, expects = get_tests()

    base_dir = spark_result
    for table_name, name in zip(table_names, names):
        target = os.path.join(base_dir, table_name, '*.csv')
        file_name = glob.glob(target)
        if os.path.isfile(file_name[0]):
            shutil.copyfile(file_name[0], os.path.join(
                arctern_result, name + '.csv'))
        else:
            print('file [%s] not exist' % file_name[0])


if __name__ == '__main__':
    collect_results()
