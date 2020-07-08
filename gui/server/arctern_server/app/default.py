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

from arctern_server.app.common import config as app_config
from arctern_server.app.common import log
from arctern_server.app import httpretry

# pylint: disable=global-statement
# pylint: disable=logging-format-interpolation

DEFAULT_INTERPRETER_TYPE = ""
DEFAULT_INTERPRETER_NAME = ""
DEFAULT_INTERPRETER_ID = ""
DEFAULT_NOTEBOOK_NAME = "arctern_default_notebook"
DEFAULT_NOTEBOOK_ID = ""
DEFAULT_PARAGRAPH_ID = ""

_python_interpreter_setting_template = {
    "name": "",
    "group": "python",
    "dependencies": [],
    "interpreterGroup": [
        {
            "class": "org.apache.zeppelin.python.PythonInterpreter",
            "defaultInterpreter": True,
            "editor": {
                "completionSupport": True,
                "editOnDblClick": True,
                "language": "python"
            },
            "name": "python"
        }
    ],
    "properties": {
        "zeppelin.python": {
            "description": "Python binary executable path. It is set to python by default.(assume python is in your $PATH)",
            "name": "zeppelin.python",
            "type": "string",
            "value": "python"
        }
    },
    "option": {
        "remote": True,
        "port": -1,
        "perNote": "shared",
        "perUser": "shared",
        "isExistingProcess": False,
        "setPermission": False,
        "owners": [],
        "isUserImpersonate": False
    }
}

_pyspark_interpreter_setting_template = {
    "name": "",
    "group": "spark",
    "properties": {
        "SPARK_HOME": {
            "name": "SPARK_HOME",
            "value": "",
            "type": "string",
            "description": "Location of spark distribution"
        },
        "master": {
            "name": "master",
            "value": "local[*]",
            "type": "string",
            "description": "Spark master uri. local | yarn-client | yarn-cluster | spark master address of standalone mode, ex) spark://master_host:7077"
        },
        "zeppelin.spark.useHiveContext": {
            "name": "zeppelin.spark.useHiveContext",
            "value": True,
            "type": "checkbox",
            "description": "Use HiveContext instead of SQLContext if it is true. Enable hive for SparkSession."
        },
        "zeppelin.spark.deprecatedMsg.show": {
            "name": "zeppelin.spark.deprecatedMsg.show",
            "value": False,
            "type": "checkbox",
            "description": "Whether show the spark deprecated message, spark 2.2 and before are deprecated. Zeppelin will display warning message by default"
        },
        "PYSPARK_PYTHON": {
            "name": "PYSPARK_PYTHON",
            "value": "python",
            "type": "string",
            "description": "Python binary executable to use for PySpark in both driver and workers (default is python2.7 if available, otherwise python). Property `spark.pyspark.python` take precedence if it is set"
        },
        "PYSPARK_DRIVER_PYTHON": {
            "name": "PYSPARK_DRIVER_PYTHON",
            "value": "python",
            "type": "string",
            "description": "Python binary executable to use for PySpark in driver only (default is `PYSPARK_PYTHON`). Property `spark.pyspark.driver.python` take precedence if it is set"
        }
    },
    "interpreterGroup": [
        {
            "name": "pyspark",
            "class": "org.apache.zeppelin.spark.PySparkInterpreter",
            "defaultInterpreter": True,
            "editor": {
                "language": "python",
                "editOnDblClick": False,
                "completionKey": "TAB",
                "completionSupport": True
            }
        },
    ],
    "dependencies": [],
    "option": {
        "remote": True,
        "port": -1,
        "perNote": "shared",
        "perUser": "shared",
        "isExistingProcess": False,
        "setPermission": False,
        "owners": [],
        "isUserImpersonate": False
    }
}

# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
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
    all_interpreter = httpretry.safe_requests("GET", interpreter_url)
    all_interpreter = all_interpreter.json()["body"]

    setting_url = app_config.ZEPPELEN_PREFIX + "/api/interpreter/setting"
    exist_setting = httpretry.safe_requests("GET", setting_url)
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
            _python_interpreter_setting_template["properties"]["zeppelin.python"]["value"] = python_path
            update_response = httpretry.safe_requests("PUT", setting_url + "/" + DEFAULT_INTERPRETER_ID, json=_python_interpreter_setting_template)
            log.INSTANCE.info(update_response.text.encode("utf-8"))
        else:       # create new interpreter setting
            _python_interpreter_setting_template["name"] = DEFAULT_INTERPRETER_NAME
            _python_interpreter_setting_template["properties"]["zeppelin.python"]["value"] = python_path
            create_response = httpretry.safe_requests("POST", setting_url, json=_python_interpreter_setting_template)
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
            _pyspark_interpreter_setting_template["name"] = DEFAULT_INTERPRETER_NAME
            _pyspark_interpreter_setting_template["properties"]["SPARK_HOME"]["value"] = spark_home
            _pyspark_interpreter_setting_template["properties"]["master"]["value"] = master_addr
            _pyspark_interpreter_setting_template["properties"]["PYSPARK_PYTHON"]["value"] = pyspark_python
            _pyspark_interpreter_setting_template["properties"]["PYSPARK_DRIVER_PYTHON"]["value"] = pyspark_driver_python
            _pyspark_interpreter_setting_template["properties"]["zeppelin.spark.useHiveContext"]["value"] = True
            _pyspark_interpreter_setting_template["properties"]["zeppelin.spark.deprecatedMsg.show"]["value"] = False
            update_response = httpretry.safe_requests("PUT", setting_url + "/" + DEFAULT_INTERPRETER_ID, json=_pyspark_interpreter_setting_template)
            log.INSTANCE.info(update_response.text.encode("utf-8"))
        else:       # create new interpreter setting
            _pyspark_interpreter_setting_template["name"] = DEFAULT_INTERPRETER_NAME
            _pyspark_interpreter_setting_template["properties"]["SPARK_HOME"]["value"] = spark_home
            _pyspark_interpreter_setting_template["properties"]["master"]["value"] = master_addr
            _pyspark_interpreter_setting_template["properties"]["PYSPARK_PYTHON"]["value"] = pyspark_python
            _pyspark_interpreter_setting_template["properties"]["PYSPARK_DRIVER_PYTHON"]["value"] = pyspark_driver_python
            _pyspark_interpreter_setting_template["properties"]["zeppelin.spark.useHiveContext"]["value"] = True
            _pyspark_interpreter_setting_template["properties"]["zeppelin.spark.deprecatedMsg.show"]["value"] = False
            create_response = httpretry.safe_requests("POST", setting_url, json=_pyspark_interpreter_setting_template)
            log.INSTANCE.info(create_response.text.encode('utf-8'))
            DEFAULT_INTERPRETER_ID = create_response.json()["body"]["id"]
        DEFAULT_INTERPRETER_NAME = "%" + DEFAULT_INTERPRETER_NAME + ".pyspark"
    else:
        raise Exception("Unsupported Interpreter Type!")
    log.INSTANCE.info("default interpreter id: {}".format(DEFAULT_INTERPRETER_ID))

def create_default_notebook():
    notebook_url = app_config.ZEPPELEN_PREFIX + "/api/notebook"
    all_notebook = httpretry.safe_requests("GET", notebook_url)
    all_notebook = all_notebook.json()["body"]
    global DEFAULT_NOTEBOOK_NAME, DEFAULT_NOTEBOOK_ID
    log.INSTANCE.info("default notebook name: {}".format(DEFAULT_NOTEBOOK_NAME))
    for notebook in all_notebook:
        if notebook["path"] == "/" + DEFAULT_NOTEBOOK_NAME:
            DEFAULT_NOTEBOOK_ID = notebook["id"]
            log.INSTANCE.info("default notebook id: {}".format(DEFAULT_NOTEBOOK_ID))
            return
    create_response = httpretry.safe_requests("POST", notebook_url, json={"name": DEFAULT_NOTEBOOK_NAME})
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
        text = DEFAULT_INTERPRETER_NAME + "\n\n"
        text += 'from arctern.util.vega import vega_choroplethmap, vega_heatmap, vega_pointmap, vega_weighted_pointmap, vega_icon, vega_fishnetmap\n'
        text += 'from arctern_pyspark import choroplethmap, heatmap, pointmap, weighted_pointmap, icon_viz, fishnetmap\n'
        text += 'from arctern_pyspark import register_funcs\n'
        text += 'register_funcs({})\n'.format("spark")
    create_response = httpretry.safe_requests("POST", paragraph_url, json={"text": text})
    log.INSTANCE.info(create_response.text.encode('utf-8'))
    DEFAULT_PARAGRAPH_ID = create_response.json()["body"]
    log.INSTANCE.info("default paragraph id: {}".format(DEFAULT_PARAGRAPH_ID))
    if DEFAULT_INTERPRETER_TYPE == "pyspark":
        from arctern_server.app.function import _function_forward
        _function_forward(text, DEFAULT_NOTEBOOK_ID, DEFAULT_PARAGRAPH_ID)
