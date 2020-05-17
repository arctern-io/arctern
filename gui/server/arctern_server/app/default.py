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
from arctern_server.app.common import config as app_config
from arctern_server.app.common import log

# pylint: disable=global-statement
# pylint: disable=logging-format-interpolation

DEFAULT_INTERPRETER_TYPE = ""
DEFAULT_INTERPRETER_NAME = ""
DEFAULT_INTERPRETER_ID = ""
DEFAULT_NOTEBOOK_NAME = "arctern_default_notebook"
DEFAULT_NOTEBOOK_ID = ""
DEFAULT_PARAGRAPH_ID = ""

# pylint: disable=too-many-statements
def create_default_interpreter():
    interpreter_type = app_config.INSTANCE.get("interpreter", "type", fallback="python")
    log.INSTANCE.info("default interpreter type: {}".format(interpreter_type))
    global DEFAULT_INTERPRETER_TYPE, DEFAULT_INTERPRETER_NAME, DEFAULT_INTERPRETER_ID
    DEFAULT_INTERPRETER_TYPE = interpreter_type
    if DEFAULT_INTERPRETER_TYPE == "python":
        default_name = "arcternpython"
    elif DEFAULT_INTERPRETER_TYPE == "pyspark":
        default_name = "arcternpyspark"
    DEFAULT_INTERPRETER_NAME = app_config.INSTANCE.get("interpreter", "name", fallback=default_name)
    log.INSTANCE.info("default interpreter name: {}".format(DEFAULT_INTERPRETER_NAME))

    interpreter_url = app_config.ZEPPELEN_PREFIX + "/api/interpreter"
    all_interpreter = requests.get(url=interpreter_url)
    all_interpreter = all_interpreter.json()["body"]

    setting_url = app_config.ZEPPELEN_PREFIX + "/api/interpreter/setting"
    exist_setting = requests.get(url=setting_url)
    exist_setting = exist_setting.json()["body"]

    exists = False
    arctern_setting = None
    for setting in exist_setting:
        if setting["name"] == DEFAULT_INTERPRETER_NAME:
            exists = True
            arctern_setting = setting
            break

    if interpreter_type == "python":
        python_path = app_config.INSTANCE.get("interpreter", "python-path", fallback="python")
        if exists:  # update interpreter setting
            if arctern_setting["group"] != "python":
                raise Exception("interpreter name already in use, please use other name instead!")
            DEFAULT_INTERPRETER_ID = arctern_setting["id"]
            arctern_setting["properties"]["zeppelin.python"]["value"] = python_path
            update_response = requests.put(url=setting_url + "/" + DEFAULT_INTERPRETER_ID, json=arctern_setting)
            log.INSTANCE.info(update_response.text.encode("utf-8"))
        else:       # create new interpreter setting
            arctern_setting = all_interpreter["python"]
            arctern_setting["name"] = DEFAULT_INTERPRETER_NAME
            arctern_setting["properties"]["zeppelin.python"]["value"] = python_path
            create_response = requests.post(url=setting_url, json=arctern_setting)
            log.INSTANCE.info(create_response.text.encode('utf-8'))
            DEFAULT_INTERPRETER_ID = create_response.json()["body"]["id"]
        DEFAULT_INTERPRETER_NAME = "%" + DEFAULT_INTERPRETER_NAME
    elif interpreter_type == "pyspark":
        spark_home = app_config.INSTANCE.get("interpreter", "spark-home", fallback="")
        master_addr = app_config.INSTANCE.get("interpreter", "master-addr", fallback="local[*]")
        pyspark_python = app_config.INSTANCE.get("interpreter", "pyspark-python", fallback="python")
        pyspark_driver_python = app_config.INSTANCE.get("interpreter", "pyspark-driver-python", fallback="python")
        if exists:  # update interpreter setting
            if arctern_setting["group"] != "spark":
                raise Exception("interpreter name already in use, please use other name instead!")
            DEFAULT_INTERPRETER_ID = arctern_setting["id"]
            arctern_setting["name"] = DEFAULT_INTERPRETER_NAME
            arctern_setting["properties"]["SPARK_HOME"]["defaultValue"] = spark_home
            arctern_setting["properties"]["master"]["defaultValue"] = master_addr
            arctern_setting["properties"]["PYSPARK_PYTHON"]["defaultValue"] = pyspark_python
            arctern_setting["properties"]["PYSPARK_DRIVER_PYTHON"]["defaultValue"] = pyspark_driver_python
            update_response = requests.put(url=setting_url + "/" + DEFAULT_INTERPRETER_ID, json=arctern_setting)
            log.INSTANCE.info(update_response.text.encode("utf-8"))
        else:       # create new interpreter setting
            arctern_setting = all_interpreter["spark"]
            arctern_setting["name"] = DEFAULT_INTERPRETER_NAME
            arctern_setting["properties"]["SPARK_HOME"]["defaultValue"] = spark_home
            arctern_setting["properties"]["master"]["defaultValue"] = master_addr
            arctern_setting["properties"]["PYSPARK_PYTHON"]["defaultValue"] = pyspark_python
            arctern_setting["properties"]["PYSPARK_DRIVER_PYTHON"]["defaultValue"] = pyspark_driver_python
            create_response = requests.post(url=setting_url, json=arctern_setting)
            log.INSTANCE.info(create_response.text.encode('utf-8'))
            DEFAULT_INTERPRETER_ID = create_response.json()["body"]["id"]
        DEFAULT_INTERPRETER_NAME = "%" + DEFAULT_INTERPRETER_NAME + ".pyspark"
    else:
        raise Exception("Unsupported Interpreter Type!")
    log.INSTANCE.info("default interpreter id: {}".format(DEFAULT_INTERPRETER_ID))

def create_default_notebook():
    notebook_url = app_config.ZEPPELEN_PREFIX + "/api/notebook"
    all_notebook = requests.get(url=notebook_url)
    all_notebook = all_notebook.json()["body"]
    global DEFAULT_NOTEBOOK_NAME, DEFAULT_NOTEBOOK_ID
    log.INSTANCE.info("default notebook name: {}".format(DEFAULT_NOTEBOOK_NAME))
    for notebook in all_notebook:
        if notebook["path"] == "/" + DEFAULT_NOTEBOOK_NAME:
            DEFAULT_NOTEBOOK_ID = notebook["id"]
            log.INSTANCE.info("default notebook id: {}".format(DEFAULT_NOTEBOOK_ID))
            return
    create_response = requests.post(url=notebook_url, json={"name": DEFAULT_NOTEBOOK_NAME})
    log.INSTANCE.info(create_response.text.encode('utf-8'))
    DEFAULT_NOTEBOOK_ID = create_response.json()["body"]
    log.INSTANCE.info("default notebook id: {}".format(DEFAULT_NOTEBOOK_ID))

def create_default_paragraph():
    paragraph_url = app_config.ZEPPELEN_PREFIX + "/api/notebook/" + DEFAULT_NOTEBOOK_ID + "/paragraph"
    global DEFAULT_PARAGRAPH_ID
    text = ""
    if DEFAULT_INTERPRETER_TYPE == "python":
        text = "%python\nprint('hello, world')\n"
    elif DEFAULT_INTERPRETER_TYPE == "pyspark":
        text = "%spark.pyspark\nprint('hello, world')\n"
    create_response = requests.post(url=paragraph_url, json={"text": text})
    log.INSTANCE.info(create_response.text.encode('utf-8'))
    DEFAULT_PARAGRAPH_ID = create_response.json()["body"]
    log.INSTANCE.info("default paragraph id: {}".format(DEFAULT_PARAGRAPH_ID))
