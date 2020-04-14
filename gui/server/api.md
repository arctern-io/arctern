### /token (not required)

获取token，该token唯一标示一个用户，且可以放到请求头中用于登录认证。

#### Request

- Method: **POST**
- URL: `/token`
- Headers: `Content-Type: application/json`
- Body:
```json
{
    "username": "zilliz",
    "password": "welcome"
}
```

#### Response

- Body
```json
{
    "code": 200,
    "data": {
        "expired": 604800,
        "token": "eyJhbGciOiJIUzUxMiIsImlhdCI6MTU4NjQ4NDM5NSwiZXhwIjoxNTg3MDg5MTk1fQ.eyJ1c2VyIjoiemlsbGl6In0.Od-3FaaiU6AkEqKaJiD3gNPCyHXyaNMIi2LkqrdpveeGZrgIH_4CNBrZ9dAQlyTYj9PFa6a1AD1vBGtViCVrvA"
    },
    "status": "success"
}
```

注意：`expired`表示token的有效时间，`token`字段将用于后续登录认证，后续用`TOKEN`指代。

### /scope

arctern的web server支持自定义命令，这里的自定义命令特指python代码，也就是说用户可以向server提交一段特定的python代码，server将执行这段代码并返回运行结果，为了使不同用户使用的变量互不影响，我们需要为这些代码创建作用域，也就是`scope`，该url用于创建一个特定的作用域。

Note: 如果后端是spark，那么创建`scope`的同时，`arctern`会为`scope`创建一个`spark`的SparkSession。随后调用的如`loadfile`、`query`、`pointmap`等restful api都会默认在spark内进行。

#### Request

- Method: **POST**
- URL: `/scope`
- Headers:
    - `Content-Type: application/json`
    - `Authorization: Token TOKEN`
- Body:
```json
{
    "scope": "scope_name"
}
```

#### Response

- 认证失败，Authorization Failed
- 认证成功，如果后台已有同名的scope，返回`error`，反之，后台新建scope，返回scope

- 注意1：后续需要认证的接口中不再列举认证失败的response，只列举认证成功的。
- 注意2：上述`request`中可以没有`scope`，也就是可以没有`body`和请求头`Content-Type: application/json`，针对这种方式的请求，后台将为该请求使用`uuid`创建一个scope，并将该`uuid`作为`scope`返回。

### /scope/<scope>

删除指定的scope，该接口一般在用户敲完所有指令后被调用。

#### Request

- Method: **DELETE**
- URL: `/scope/<scope>`
- Headers:
    - `Content-Type: application/json`
    - `Authorization: Token TOKEN`

#### Response

- `scope`不存在，删除失败。
- `scope`存在，删除指定的作用域。

### /command

在指定作用域(`scope`)内执行python代码，后台返回执行结果。

#### Request

- Method: **POST**
- URL: `/command`
- Headers:
    - `Content-Type: application/json`
    - `Authorization: Token TOKEN`
- Body:
```json
{
    "scope": "scope_name",
    "command": "import sys\nlen(sys.argv)"
}
```

#### Response

- `scope`不存在，报错，提示作用域不存在。
- 在指定`scope`内执行对应的`command`，并收集运行结果：
    - 运行成功，返回运行结果。
    - 出现异常，返回异常信息以及错误码400。

### /session (not required)

在指定作用域(`scope`)内创建`SparkSession`，创建`SparkSession`的过程由server自动完成，`SparkSession`可用于后续建表、执行sql、画图等spark任务。

#### Request

- Method: **POST**
- URL: `/session`
- Headers:
    - `Content-Type: application/json`
    - `Authorization: Token TOKEN`
- Body:
```json
{
    "scope->scope"(scope): "scope_name"(can be default),
    "session->session"(session): "arctern"(can be default),
    "app_name": "arctern",
    "master-addr": "local[*]",
    "configs":{
        "spark.executorEnv.GDAL_DATA": "",
        "spark.executorEnv.PROJ_LIB": ""
    },
    "envs": {
    }
}
```

注意：`body`中两个必要的字段是`scope`以及`session`，`app_name`若未指定则由后台生成唯一的spark应用名，`master-addr`表示spark的master节点，目前支持`yarn`、`standalone`、`local`等spark部署模式，若未指定则默认为`local[*]`，`configs`字段为创建`SparkSession`时需要进行的配置，后台内部有一些必要的默认配置，`envs`表示创建`SparkSession`时需要对系统环境变量进行的配置。

#### Response

- `scope`不存在，报错，提示作用域不存在。
- `scope`内已有同名的session，报错，提示同名冲突。
- 在指定scope内，根据请求参数创建对应的`SparkSession`：
    - 成功创建，提示成功创建。
    - 出现异常，返回异常信息以及错误码400。

### /loadfile

在指定作用域(`scope`)内使用指定的`SparkSession`加载文件内容，并执行对应的建表操作。

#### Request

- Method: **POST**
- URL: `/loadfile`
- Headers:
    - `Content-Type: application/json`
    - `Authorization: Token TOKEN`
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

目前支持的文件类型有，即`format`字段的可选参数有：
- csv
- json
- parquet

session：该参数optional，默认为`spark`。
schema：各列的schema描述，`schema`字段是一个list，顺序要和文件中各列的实际存储顺序相同。当前支持的schema类型有：TODO



#### Response

- `scope`不存在，报错，提示作用域不存在。
- `session`不存在，报错，提示没有对应的SparkSession。
- 加载数据时出现异常，返回异常信息以及错误码400。
- 加载数据成功，提示建表成功。

### /createtable (not required)

从已有的数据表创建新的表，将sql的结果作为要创建的表。

- Method: **POST**
- URL: /createtable
- Headers:
    - `Content-Type: application/json`
    - `Authorization: Token TOKEN`
- Body:
```json
{
    "scope": "scope_name",
    "session": "arctern",
    "tables": [
        {
            "name": "nyc_taxi",
            "sql": "select VendorID, to_timestamp(tpep_pickup_datetime,'yyyy-MM-dd HH:mm:ss XXXXX') as tpep_pickup_datetime, to_timestamp(tpep_dropoff_datetime,'yyyy-MM-dd HH:mm:ss XXXXX') as tpep_dropoff_datetime, passenger_count, trip_distance, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, fare_amount, tip_amount, total_amount, buildingid_pickup, buildingid_dropoff, buildingtext_pickup, buildingtext_dropoff from old_nyc_taxi where (pickup_longitude between -180 and 180) and (pickup_latitude between -90 and 90) and (dropoff_longitude between -180 and 180) and  (dropoff_latitude between -90 and 90)"
        }
    ]
}
```

#### Response

- `scope`不存在，报错，提示作用域不存在。
- `session`不存在，报错，提示没有对应的SparkSession。
- 执行sql语句时出现异常，返回异常信息以及错误码400。
- 执行sql语句成功，将sql结果作为要创建的表，提示建表成功。

### /table/<scope>/<session>/<table_name>

#### Request

- Method: **GET**
- URL: /table/<scope>/<session>/<table_name>
- Headers:
    - `Authorization: Token TOKEN`

#### Response

以`/table/scope_1/arctern/nyc_taxi`为例，该接口返回`nyc_taxi`这张表的`schema`信息。

```json
{
    "schema": [
        {"column0": "string"},
        {"column1": "double"},
        {"column2": "int"}
    ],
    "num_rows": 500
}
```

TODO: 对比pyspark和pandas的交集。

### /query

执行sql语句，将sql结果用json形式返回。

- Method: **POST**
- URL: `/query`
- Headers:
    - `Content-Type: application/json`
    - `Authorization: Token TOKEN`
- Body:
```json
{
    "scope": "scope_name",
    "session": "arctern",
    "sql": "select fare_amount from nyc_taxi limit 1"
}
```

```json
{
    "scope": "scope_name",
    "session": "arctern",
    "sql": "create table new_table as (select * from nyc_taxi)"
}
```

#### Response

- `scope`不存在，报错，提示作用域不存在。
- `session`不存在，报错，提示没有对应的SparkSession。
- 执行对应sql语句出现异常，返回异常信息以及错误码400
- 执行成功，body如下
```json
{
    "status": "success",
    "code": 200,
    "result": [
        2.5
    ]
}
```

### /pointmap

点图，根据sql语句以及相关画图参数绘制点图，将编码后图片数据返回。

- Method: **POST**
- URL: `/pointmap`
- Headers:
    - `Content-Type: application/json`
    - `Authorization: Token TOKEN`
- Body:
```json
{
    "scope": "scope_name",
    "session": "arctern",
    "sql": "",
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

TODO: sql结果要符合画图函数的输入约定。

#### Response

- `scope`不存在，报错，提示作用域不存在。
- `session`不存在，报错，提示没有对应的SparkSession。
- 执行对应sql语句出现异常，返回异常信息以及错误码400
- 执行成功，body如下
```json
{
    "status": "success",
    "code": 200,
    "result": "image data encoded with base64"
}
```

### /heatmap

热力图，根据sql语句以及相关画图参数绘制热力图，将编码后图片数据返回。

- Method: **POST**
- URL: `/heatmap`
- Headers:
    - `Content-Type: application/json`
    - `Authorization: Token TOKEN`
- Body
```json
{
    "scope": "scope_1",
    "session": "arctern",
    "sql": "select ST_Point(pickup_longitude, pickup_latitude) as point, passenger_count as w from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude), ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))",
    "params": {
        "width": 1024,
        "height": 896,
        "bounding_box": [-75.37976, 40.191296, -71.714099, 41.897445],
        "coordinate_system": "EPSG:4326",
        "map_zoom_level": 10
    }
}
```

#### Response

- `scope`不存在，报错，提示作用域不存在。
- `session`不存在，报错，提示没有对应的SparkSession。
- 执行对应sql语句出现异常，返回异常信息以及错误码400
- 执行成功，body如下
```json
{
    "status": "success",
    "code": 200,
    "result": "image data encoded with base64"
}
```

### /choropleth

热力图，根据sql语句以及相关画图参数绘制轮廓图，将编码后图片数据返回。

- Method: **POST**
- URL: `/choropleth`
- Headers:
    - `Content-Type: application/json`
    - `Authorization: Token TOKEN`
- Body
```json
{
        "scope": "scope_name",
        "session": "arctern",
        "sql": "select ST_GeomFromText(buildingtext_dropoff) as wkt, passenger_count as w from nyc_taxi",
        "params": {
            "width": 1024,
            "height": 896,
            "bounding_box": [-75.37976, 40.191296, -71.714099, 41.897445],
            "coordinate_system": "EPSG:4326",
            "color_gradient": ["#0000FF", "#FF0000"],
            "color_bound": [2.5, 5],
            "opacity": 1
        }
    }
```

#### Response

- `scope`不存在，报错，提示作用域不存在。
- `session`不存在，报错，提示没有对应的SparkSession。
- 执行对应sql语句出现异常，返回异常信息以及错误码400
- 执行成功，body如下
```json
{
    "status": "success",
    "code": 200,
    "result": "image data encoded with base64"
}
```

### /weighted

热力图，根据sql语句以及相关画图参数绘制权重图，将编码后图片数据返回。

- Method: **POST**
- URL: `/weighted`
- Headers:
    - `Content-Type: application/json`
    - `Authorization: Token TOKEN`
- Body
```json
{
        "scope": "scope_name",
        "session": "arctern",
        "sql": "select ST_Point(pickup_longitude, pickup_latitude) as point, tip_amount as c, fare_amount as s from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude), ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))",
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

#### Response

- `scope`不存在，报错，提示作用域不存在。
- `session`不存在，报错，提示没有对应的SparkSession。
- 执行对应sql语句出现异常，返回异常信息以及错误码400
- 执行成功，body如下
```json
{
    "status": "success",
    "code": 200,
    "result": "image data encoded with base64"
}
```
