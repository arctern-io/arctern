### 创建scope

arctern的web server支持自定义命令，这里的自定义命令特指python代码，也就是说用户可以向server提交一段特定的python代码，server将执行这段代码并返回运行结果，为了使不同用户使用的变量互不影响，我们需要为这些代码创建作用域，也就是`scope`，该url用于创建一个特定的作用域。

Note: 如果后端是`spark`，那么创建`scope`的同时，`arctern`会为`scope`创建一个叫做`spark`的`SparkSession`。随后调用的如`/loadfile`、`/query`、`/pointmap`等restful api都会默认在`spark`内进行。

#### Request

- Method: **POST**
- URL: `/scope`
- Headers:
    - `Content-Type: application/json`
- Body:
```json
{
    "scope": "scope_name"
}
```

参数说明：

- scope：可选参数，若不指定`scope`，则request的`header`可以省略`Content-Type: application/json`，且后台使用`uuid`作为新建的`scope`。

样例：

```python
import requests

url = "http://localhost:8080/scope"

payload = "{\n\t\"scope\":\"scope_name\"\n}"
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data = payload)

print(response.text.encode('utf8'))
```

```shell
curl --location --request POST 'http://localhost:8080/scope' \
--header 'Content-Type: application/json' \
--data-raw '{
	"scope":"scope_name"
}'
```

#### Response

成功样例：

```json
{
    "status": "success",
    "code": "200",
    "message": "create scope successfully!",
    "scope": "scope_name"
}
```

错误样例：

```json
{
    "status": "error",
    "code": "-1",
    "message": "scope already exist!"
}
```

### 删除scope

删除指定的`scope`，该接口一般在用户敲完所有指令后被调用。

#### Request

- Method: **DELETE**
- URL: `/scope/<scope>`
- Headers:
    - `Content-Type: application/json`

样例：

```python
import requests

url = "http://localhost:8080/scope/scope_name"

payload = {}
headers= {}

response = requests.request("DELETE", url, headers=headers, data = payload)

print(response.text.encode('utf8'))
```

```shell
curl --location --request DELETE 'http://localhost:8080/scope/scope_name'
```

#### Response

成功样例：

```json
{
    "status": "success",
    "code": "200",
    "message": "delete scope successfully!",
    "scope": "scope_name"
}
```

错误样例：

```json
{
    "status": "error",
    "code": "-1",
    "message": "scope not found!"
}
```

### 执行命令

在指定作用域内(`scope`)执行python代码，后台返回执行结果。

#### Request

- Method: **POST**
- URL: `/command`
- Headers:
    - `Content-Type: application/json`
- Body:
```json
{
    "scope": "scope_name",
    "command": "import sys\nprint(len(sys.argv))"
}
```

参数说明：

- scope：该字段指明在哪一个作用域内执行command；
- command：待执行的`python`代码。

样例：

```python
import requests

url = "http://localhost:8080/command"

payload = "{\n\t\"scope\":\"scope_name\",\n\t\"comamnd\":\"import sys\\nprint(len(sys.argv))\"\n}"
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data = payload)

print(response.text.encode('utf8'))
```

```shell
curl --location --request POST 'http://localhost:8080/command' \
--header 'Content-Type: application/json' \
--data-raw '{
	"scope":"scope_name",
	"comamnd":"import sys\nprint(len(sys.argv))"
}'
```

#### Response

正常执行：

```json
{
    "status": "success",
    "code": "200",
    "message": "execute command successfully!"
}
```

执行对应代码出现异常：

```json
{
    "status": "error",
    "code": "400",
    "message": "cannot import package1"
}
```

### 从文件中加载表

在指定作用域(`scope`)内加载文件内容，并执行对应的建表操作，注意，该操作使用`scope`内的`spark`作为默认的`SparkSession`，详见[/scope](###/scope)。

#### Request

- Method: **POST**
- URL: `/loadfile`
- Headers:
    - `Content-Type: application/json`
- Body:
```json
{
    "scope": "scope_name",
    "session": "session_name", 
    "tables": [
        {
            "name": "table_name",
            "format": "csv",
            "path": "/path/to/data.csv",
            "options": {
                "header": "True",
                "delimiter": ","
            },
            "schema": [
                {"column0": "string"},
                {"column1": "double"},
                {"column2": "int"}
            ]
        }
    ]
}
```

参数说明：

- scope：该字段指明在哪一个作用域内执行建表操作；
- session：可选参数，该字段指明使用哪个`SparkSession`执行建表操作，若未指定，则使用默认的`spark`；
- tables：建表描述，该字段为一个列表(`list`)，后台会按顺序进行建表操作，以下为具体参数说明：
    - name：表名；
    - format：待加载文件的文件格式，目前支持的文件格式有：
        - csv
        - json
        - parquet
    - path：文件路径；
    - options：加载文件时需要指定的选项，该字段使用`key-value`形式，和spark读取文件时保持一致；
    - schema：各列的schema描述，schema字段是一个列表(`list`)，顺序要和文件中各列的实际存储顺序相同。当前支持的schema类型有：（TODO）

样例：

```python
import requests

url = "http://localhost:8080/loadfile"

payload = "{\n    \"scope\": \"scope_name\",\n    \"session\": \"session_name\",\n    \"tables\": [\n        {\n            \"name\": \"table_name\",\n            \"format\": \"csv\",\n            \"path\": \"/path/to/data.csv\",\n            \"options\": {\n                \"header\": \"True\",\n                \"delimiter\": \",\"\n            },\n            \"schema\": [\n                {\n                    \"column0\": \"string\"\n                },\n                {\n                    \"column1\": \"double\"\n                },\n                {\n                    \"column2\": \"int\"\n                }\n            ]\n        }\n    ]\n}"
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data = payload)

print(response.text.encode('utf8'))
```

```shell
curl --location --request POST 'http://localhost:8080/loadfile' \
--header 'Content-Type: application/json' \
--data-raw '{
    "scope": "scope_name",
    "session": "session_name",
    "tables": [
        {
            "name": "table_name",
            "format": "csv",
            "path": "/path/to/data.csv",
            "options": {
                "header": "True",
                "delimiter": ","
            },
            "schema": [
                {
                    "column0": "string"
                },
                {
                    "column1": "double"
                },
                {
                    "column2": "int"
                }
            ]
        }
    ]
}'
```

### Response

成功样例：

```json
{
    "status": "success",
    "code": "200",
    "message": "create table successfully!"
}
```

错误样例：

```json
{
    "status": "error",
    "code": "-1",
    "message": "scope not found!"
}
```

### 保存到文件

在指定作用域(`scope`)内执行表数据写盘操作，将`sql`的结果存入指定文件，注意，该操作使用`scope`内的`spark`作为默认的`SparkSession`，详见[/scope](###/scope)。

#### Request

- Method: **POST**
- URL: `/savetable`
- Headers:
    - `Content-Type: application/json`
- Body:
```json
{
    "scope": "scope_name",
    "session": "session_name", 
    "tables": [
        {
            "sql": "select * from table_name",
            "format": "csv",
            "path": "/path/to/data.csv",
            "options": {
                "header": "True",
                "delimiter": ","
            }
        }
    ]
}
```

参数说明：

- scope：该字段指明在哪一个作用域内执行建表操作；
- session：可选参数，该字段指明使用哪个`SparkSession`执行建表操作，若未指定，则使用默认的`spark`；
- tables：写盘操作描述，该字段为一个列表(`list`)，后台会按顺序进行写盘操作，以下为具体参数说明：
    - sql：待执行的query语句，该语句的结果将作为要保存的表；
    - format：待加载文件的文件格式，目前支持的文件格式有：
        - csv
        - json
        - parquet
    - path：文件路径；
    - options：保存文件时需要指定的选项，该字段使用`key-value`形式，和spark保存文件时保持一致；

样例：

```python
import requests

url = "http://localhost:8080/savetable"

payload = "```json\n{\n    \"scope\": \"scope_name\",\n    \"session\": \"session_name\", \n    \"tables\": [\n        {\n            \"sql\": \"select * from table_name\",\n            \"format\": \"csv\",\n            \"path\": \"/path/to/data.csv\",\n            \"options\": {\n                \"header\": \"True\",\n                \"delimiter\": \",\"\n            }\n        }\n    ]\n}"
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data = payload)

print(response.text.encode('utf8'))
```

```shell
curl --location --request POST 'http://localhost:8080/savetable' \
--header 'Content-Type: application/json' \
--data-raw '```json
{
    "scope": "scope_name",
    "session": "session_name", 
    "tables": [
        {
            "sql": "select * from table_name",
            "format": "csv",
            "path": "/path/to/data.csv",
            "options": {
                "header": "True",
                "delimiter": ","
            }
        }
    ]
}'
```

### Response

成功样例：

```json
{
    "status": "success",
    "code": "200",
    "message": "save table successfully!"
}
```

错误样例：

```json
{
    "status": "error",
    "code": "-1",
    "message": "scope not found!"
}
```

### 表信息统计

返回表的统计信息，包括`schema`信息、行数等。(TODO: spark与pandas交集)

#### Request

- Method: **GET**
- URL: /table?scope=scope1&session=spark&table=table1

- scope_name：该字段指明在哪一个作用域内查询表的信息；
- session_name：可选参数，该字段指明使用哪个`SparkSession`查询表的信息，若未指定，则使用默认的`spark`；
- table_name：表名。

样例：

```python
import requests

url = "http://localhost:8080/table?scope=scope1&session=spark&table=table1"

payload = {}
headers= {}

response = requests.request("GET", url, headers=headers, data = payload)

print(response.text.encode('utf8'))
```

```shell
curl --location --request GET 'http://localhost:8080/table?scope=scope1&session=spark&table=table1'
```

### Response

成功样例：

```json
{
    "table": "table_name",
    "schema": [
        {"column0": "string"},
        {"column1": "double"},
        {"column2": "int"}
    ],
    "num_rows": 500
}
```

错误样例：

```json
{
    "status": "error",
    "code": "-1",
    "message": "table not found!"
}
```

### 查询

执行`sql`语句，可以选择是否将`sql`语句的结果返回，如果是，则将结果以`json`形式返回。

#### Request

- Method: **POST**
- URL: `/query`
- Headers:
    - `Content-Type: application/json`
- Body:
```json
{
    "scope": "scope_name",
    "session": "session_name",
    "sql": "select * from table_name limit 1",
    "collect_result": "1"
}
```

```json
{
    "scope": "scope_name",
    "session": "session_name",
    "sql": "create table new_table as (select * from table_name)",
    "collect_result": "0"
}
```

参数说明：

- scope：该字段指明在哪一个作用域内执行`sql`语句；
- session：可选参数，该字段指明使用哪个`SparkSession`执行`sql`语句，若未指定，则使用默认的`spark`；
- sql：待执行的query语句；
- collect_result：可选参数，默认为`1`，`1`表示将`sql`语句的结果用`json`形式返回，`0`表示不返回结果。

样例：

```python
import requests

url = "http://localhost:8080/query"

payload = "{\n    \"scope\": \"scope_name\",\n    \"session\": \"session_name\",\n    \"sql\": \"select * from table_name limit 1\",\n    \"collect_result\": \"1\"\n}"
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data = payload)

print(response.text.encode('utf8'))
```

```shell
curl --location --request POST 'http://localhost:8080/query' \
--header 'Content-Type: application/json' \
--data-raw '{
    "scope": "scope_name",
    "session": "session_name",
    "sql": "select * from table_name limit 1",
    "collect_result": "1"
}'
```

```python
import requests

url = "http://localhost:8080/query"

payload = "{\n    \"scope\": \"scope_name\",\n    \"session\": \"session_name\",\n    \"sql\": \"create table new_table as (select * from table_name)\",\n    \"collect_result\": \"0\"\n}"
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data = payload)

print(response.text.encode('utf8'))
```

```shell
curl --location --request POST 'http://localhost:8080/query' \
--header 'Content-Type: application/json' \
--data-raw '{
    "scope": "scope_name",
    "session": "session_name",
    "sql": "create table new_table as (select * from table_name)",
    "collect_result": "0"
}'
```

#### Response

样例：

```json
{
    "status": "succuss",
    "code": "200",
    "result": [
        ["col1", 0.5, 2]
    ],
    "message": "execute sql successfully!"
}
```

### 点图

点图，根据`sql`语句以及相关画图参数绘制点图，将编码后的图片数据返回。

- Method: **POST**
- URL: `/pointmap`
- Headers:
    - `Content-Type: application/json`
- Body:
```json
{
    "scope": "scope_name",
    "session": "session_name",
    "sql": "select ST_Point(col2, col2) as point from table_name",
    "params": {
        "width": 1024,
        "height": 896,
        "bounding_box": [-75.37976, 40.191296, -71.714099, 41.897445],
        "coordinate_system": "EPSG:4326",
        "point_color": "#2DEF4A",
        "point_size": 3,
        "opacity": 0.5
    }
}
```

参数说明：

- scope：该字段指明在哪一个作用域内执行`sql`语句；
- session：可选参数，该字段指明使用哪个`SparkSession`执行`sql`语句，若未指定，则使用默认的`spark`；
- sql：待执行的query语句，该sql的结果作为绘制点图的渲染对象；
- params：绘制点图时需要的参数，具体说明如下：
    - width：图片宽度；
    - height：图片高度；
    - bounding_box：渲染图片所表示的地理范围[`x_min`, `y_min`, `x_max`, `y_max`]；
    - coordinate_system：输入数据的坐标系统，详见[World Geodetic System](https://en.wikipedia.org/wiki/World_Geodetic_System)；
    - point_size：点的大小；
    - point_color：点的颜色；
    - opacity：点的不透明度；

样例：

```python
import requests

url = "http://localhost:8080/pointmap"

payload = "{\n    \"scope\": \"scope_name\",\n    \"session\": \"session_name\",\n    \"sql\": \"select ST_Point(col2, col2) as point from table_name\",\n    \"params\": {\n        \"width\": 1024,\n        \"height\": 896,\n        \"bounding_box\": [-75.37976, 40.191296, -71.714099, 41.897445],\n        \"coordinate_system\": \"EPSG:4326\",\n        \"point_color\": \"#2DEF4A\",\n        \"point_size\": 3,\n        \"opacity\": 0.5\n    }\n}"
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data = payload)

print(response.text.encode('utf8'))
```

```shell
curl --location --request POST 'http://localhost:8080/pointmap' \
--header 'Content-Type: application/json' \
--data-raw '{
    "scope": "scope_name",
    "session": "session_name",
    "sql": "select ST_Point(col2, col2) as point from table_name",
    "params": {
        "width": 1024,
        "height": 896,
        "bounding_box": [-75.37976, 40.191296, -71.714099, 41.897445],
        "coordinate_system": "EPSG:4326",
        "point_color": "#2DEF4A",
        "point_size": 3,
        "opacity": 0.5
    }
}'
```

#### Response

样例：

```json
{
    "status": "success",
    "code": "200",
    "result": "使用base64编码后的点图数据"
}
```

### 热力图

热力图，根据`sql`语句以及相关画图参数绘制热力图，将编码后的图片数据返回。

- Method: **POST**
- URL: `/heatmap`
- Headers:
    - `Content-Type: application/json`
- Body:
```json
{
    "scope": "scope_name",
    "session": "session_name",
    "sql": "select ST_Point(col2, col2) as point from table_name",
    "params": {
        "width": 1024,
        "height": 896,
        "bounding_box": [-75.37976, 40.191296, -71.714099, 41.897445],
        "coordinate_system": "EPSG:4326",
        "map_zoom_level": 10,
        "aggragation_type": "sum"
    }
}
```

参数说明：

- scope：该字段指明在哪一个作用域内执行`sql`语句；
- session：可选参数，该字段指明使用哪个`SparkSession`执行`sql`语句，若未指定，则使用默认的`spark`；
- sql：待执行的query语句，该sql的结果作为绘制热力图的渲染对象；
- params：绘制热力图时需要的参数，具体说明如下：
    - width：图片宽度；
    - height：图片高度；
    - bounding_box：渲染图片所表示的地理范围[`x_min`, `y_min`, `x_max`, `y_max`]；
    - coordinate_system：输入数据的坐标系统，详见[World Geodetic System](https://en.wikipedia.org/wiki/World_Geodetic_System)；
    - map_zoom_level：地图放大比例，取值范围`(1 ~ 15)`；
    - aggregation_type：聚合类型。

样例：

```python
import requests

url = "http://localhost:8080/heatmap"

payload = "{\n    \"scope\": \"scope_name\",\n    \"session\": \"session_name\",\n    \"sql\": \"select ST_Point(col2, col2) as point from table_name\",\n    \"params\": {\n        \"width\": 1024,\n        \"height\": 896,\n        \"bounding_box\": [-75.37976, 40.191296, -71.714099, 41.897445],\n        \"coordinate_system\": \"EPSG:4326\",\n        \"map_zoom_level\": 10\n    }\n}"
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data = payload)

print(response.text.encode('utf8'))
```

```shell
curl --location --request POST 'http://localhost:8080/heatmap' \
--header 'Content-Type: application/json' \
--data-raw '{
    "scope": "scope_name",
    "session": "session_name",
    "sql": "select ST_Point(col2, col2) as point from table_name",
    "params": {
        "width": 1024,
        "height": 896,
        "bounding_box": [-75.37976, 40.191296, -71.714099, 41.897445],
        "coordinate_system": "EPSG:4326",
        "map_zoom_level": 10
    }
}'
```

#### Response

样例：

```json
{
    "status": "success",
    "code": "200",
    "result": "使用base64编码后的热力图数据"
}
```

### 轮廓图

轮廓图，根据`sql`语句以及相关画图参数绘制轮廓图，将编码后的图片数据返回。

- Method: **POST**
- URL: `/choroplethmap`
- Headers:
    - `Content-Type: application/json`
- Body:
```json
{
    "scope": "scope_name",
    "session": "session_name",
    "sql": "select ST_Point(col2, col2) as point, col2 as count from table_name",
    "params": {
        "width": 1024,
        "height": 896,
        "bounding_box": [-75.37976, 40.191296, -71.714099, 41.897445],
        "coordinate_system": "EPSG:4326",
        "color_gradient": ["#0000FF", "#FF0000"],
        "color_bound": [2.5, 5],
        "opacity": 1,
        "aggregation_type": "sum"
    }
}
```

参数说明：

- scope：该字段指明在哪一个作用域内执行`sql`语句；
- session：可选参数，该字段指明使用哪个`SparkSession`执行`sql`语句，若未指定，则使用默认的`spark`；
- sql：待执行的query语句，该sql的结果作为绘制轮廓图的渲染对象；
- params：绘制轮廓图时需要的参数，具体说明如下：
    - width：图片宽度；
    - height：图片高度；
    - bounding_box：渲染图片所表示的地理范围[`x_min`, `y_min`, `x_max`, `y_max`]；
    - coordinate_system：输入数据的坐标系统，详见[World Geodetic System](https://en.wikipedia.org/wiki/World_Geodetic_System)；
    - color_gradient：点的颜色渐变范围，即点的颜色从左边渐变到右边；
    - color_bound：点颜色的取值范围，与`color_gradient`配合使用；
    - opacity：点的不透明度。
    - aggregation_type：聚合类型。

样例：

```python
import requests

url = "http://localhost:8080/choroplethmap"

payload = "{\n    \"scope\": \"scope_name\",\n    \"session\": \"session_name\",\n    \"sql\": \"select ST_Point(col2, col2) as point, col2 as count from table_name\",\n    \"params\": {\n        \"width\": 1024,\n        \"height\": 896,\n        \"bounding_box\": [-75.37976, 40.191296, -71.714099, 41.897445],\n        \"coordinate_system\": \"EPSG:4326\",\n        \"color_gradient\": [\"#0000FF\", \"#FF0000\"],\n        \"color_bound\": [2.5, 5],\n        \"opacity\": 1\n    }\n}"
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data = payload)

print(response.text.encode('utf8'))
```

```shell
curl --location --request POST 'http://localhost:8080/choroplethmap' \
--header 'Content-Type: application/json' \
--data-raw '{
    "scope": "scope_name",
    "session": "session_name",
    "sql": "select ST_Point(col2, col2) as point, col2 as count from table_name",
    "params": {
        "width": 1024,
        "height": 896,
        "bounding_box": [-75.37976, 40.191296, -71.714099, 41.897445],
        "coordinate_system": "EPSG:4326",
        "color_gradient": ["#0000FF", "#FF0000"],
        "color_bound": [2.5, 5],
        "opacity": 1
    }
}'
```

#### Response

样例：

```json
{
    "status": "success",
    "code": "200",
    "result": "使用base64编码后的轮廓图数据"
}
```

### 权重图

权重图，根据`sql`语句以及相关画图参数绘制权重图，将编码后的图片数据返回。

- Method: **POST**
- URL: `/weighted_pointmap`
- Headers:
    - `Content-Type: application/json`
- Body:
```json
{
    "scope": "scope_name",
    "session": "session_name",
    "sql": "select ST_Point(col2, col2) as point, col2 as count1, col2 as count2 from table_name",
    "params": {
            "width": 1024,
            "height": 896,
            "bounding_box": [-75.37976, 40.191296, -71.714099, 41.897445],
            "color_gradient": ["#0000FF", "#FF0000"],
            "color_bound": [0, 2],
            "size_bound": [0, 10],
            "opacity": 1.0,
            "coordinate_system": "EPSG:4326"
    }
}
```

参数说明：

- scope：该字段指明在哪一个作用域内执行`sql`语句；
- session：可选参数，该字段指明使用哪个`SparkSession`执行`sql`语句，若未指定，则使用默认的`spark`；
- sql：待执行的query语句，该sql的结果作为绘制权重图的渲染对象；
- params：绘制权重图时需要的参数，具体说明如下：
    - width：图片宽度；
    - height：图片高度；
    - bounding_box：渲染图片所表示的地理范围[`x_min`, `y_min`, `x_max`, `y_max`]；
    - coordinate_system：输入数据的坐标系统，详见[World Geodetic System](https://en.wikipedia.org/wiki/World_Geodetic_System)；
    - color_gradient：点的颜色渐变范围，即点的颜色从左边渐变到右边；
    - color_bound：点颜色的取值范围，与`color_gradient`配合使用；
    - opacity：点的不透明度；
    - size_bound：点大小的取值范围。

样例：

```python
import requests

url = "http://localhost:8080/weighted_pointmap"

payload = "{\n    \"scope\": \"scope_name\",\n    \"session\": \"session_name\",\n    \"sql\": \"select ST_Point(col2, col2) as point, col2 as count1, col2 as count2 from table_name\",\n    \"params\": {\n            \"width\": 1024,\n            \"height\": 896,\n            \"bounding_box\": [-75.37976, 40.191296, -71.714099, 41.897445],\n            \"color_gradient\": [\"#0000FF\", \"#FF0000\"],\n            \"color_bound\": [0, 2],\n            \"size_bound\": [0, 10],\n            \"opacity\": 1.0,\n            \"coordinate_system\": \"EPSG:4326\"\n    }\n}"
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data = payload)

print(response.text.encode('utf8'))
```

```shell
curl --location --request POST 'http://localhost:8080/weighted_pointmap' \
--header 'Content-Type: application/json' \
--data-raw '{
    "scope": "scope_name",
    "session": "session_name",
    "sql": "select ST_Point(col2, col2) as point, col2 as count1, col2 as count2 from table_name",
    "params": {
            "width": 1024,
            "height": 896,
            "bounding_box": [-75.37976, 40.191296, -71.714099, 41.897445],
            "color_gradient": ["#0000FF", "#FF0000"],
            "color_bound": [0, 2],
            "size_bound": [0, 10],
            "opacity": 1.0,
            "coordinate_system": "EPSG:4326"
    }
}'
```

#### Response

样例：

```json
{
    "status": "success",
    "code": "200",
    "result": "使用base64编码后的权重图数据"
}
```


