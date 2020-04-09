# arctern-gui-server

一个简易的RESTFUL接口的web服务器，采用python语言编写，主要是为了使用spark服务。

## 代码结构

## 构建环境

构建conda环境
conda create -n arctern python=3.7

进入arctern虚拟环境
conda activate arctern

安装arctern包
conda install -y -q -c conda-forge -c arctern-dev arctern-spark

进入项目目录，运行下面的命令构建环境

```bash
pip install -r requirements.txt
```

安装pyspark，推荐安装spark版本自带的
进入spark根目录，执行如下命令

```bash
cd python
python setup.py install
```

## 下载测试数据

在`https://github.com/zilliztech/arctern-tutorial/tree/master/data`下获取0_5M_nyc_taxi_and_building.csv
保持文件不变，放在data目录下

## 启动web服务

运行一下命令可以启动web服务

```bash
python manage.py -r
```

其中命令行参数说明如下：
-h help
-r production mode
-p http port
-i http ip
-c [path/to/data-config] load data

服务器启动后，可以通过[Load API](#api_load_part)可以动态加载数据

如果期望服务器启动，自动加载数据

```bash
python manage.py -r -c path/to/db.json
```

其中，db.json内容的格式在[这里](#api_load_part)

- 拷贝request.json对应的json内容，保存为db.json
- 修改对应的参数项

## 与spark服务对接

### spark local mode

需要设置 spark.executorEnv.PYSPARK_PYTHON

### spark standalone mode

需要web服务上与spark服务器具有相同的配置环境

### spark hadoop/yarn mode

## 通过curl测试

### Get接口测试

```bash
curl http://127.0.0.1:5000/api/gis/func1
//带参数
curl http://127.0.0.1:5000/api/gis/func1?a=1&b=2
```

### Post接口测试

```bash
curl -X POST  -d '{"a":1}' -H "Content-Type: application/json" http://127.0.0.1:5000/api/gis/func2
```

## 接口定义

### token设置

```shell
-H "Authorization: Token <jws-token>"
```

### 错误说明

如果服务器出现逻辑错误，当前统一返回

```json
response:
    {
        "status": "error",
        "code": -1,
        "message": "description"
    }
```

### /login 登陆

method: POST
token：NO

```json
request:
    {
        "username": "arctern",   //用户名
        "password": "*******"    // 密码
    }
response:
    {
        "status": "success",
        "code": 200,
        "data": {
            "token": "xxxxx",       //token
            "expired": "122222"     //有效时间，秒数
        },
        "message": null
    }
```

for example:

```shell
curl -X POST -H "Content-Type: application/json" -d '{"username":"zilliz", "password":"123456"}' http://127.0.0.1:8080/login
```

### <span id="api_load_part"/> /load加载表数据

method: POST
token: yes

`request.json`:

```json
{
    "db_name": "db1",
    "type": "spark",
    "spark": {
        "app_name": "arctern",
        "master-addr": "local[*]",
        "envs": {
            "PYSPARK_PYTHON": "/home/ljq/miniconda3/envs/zgis_dev/bin/python"
        },
        "configs": {
            "spark.sql.execution.arrow.pyspark.enabled": "true",
            "spark.databricks.session.share": "false"
        }
    },
    "tables": [
        {
            "name": "old_nyc_taxi",
            "format": "csv",
            "path": "/home/ljq/work/arctern/gui/server/data/0_5M_nyc_taxi_and_building.csv",
            "options": {
                "header": "True",
                "delimiter": ","
            },
            "schema": [
                {"VendorID": "string"},
                {"tpep_pickup_datetime": "string"},
                {"tpep_dropoff_datetime": "string"},
                {"passenger_count": "long"},
                {"trip_distance": "double"},
                {"pickup_longitude": "double"},
                {"pickup_latitude": "double"},
                {"dropoff_longitude": "double"},
                {"dropoff_latitude": "double"},
                {"fare_amount": "double"},
                {"tip_amount": "double"},
                {"total_amount": "double"},
                {"buildingid_pickup": "long"},
                {"buildingid_dropoff": "long"},
                {"buildingtext_pickup": "string"},
                {"buildingtext_dropoff": "string"}
            ],
            "visibility": "False"
        },
        {
            "name": "nyc_taxi",
            "sql": "select VendorID, to_timestamp(tpep_pickup_datetime,'yyyy-MM-dd HH:mm:ss XXXXX') as tpep_pickup_datetime, to_timestamp(tpep_dropoff_datetime,'yyyy-MM-dd HH:mm:ss XXXXX') as tpep_dropoff_datetime, passenger_count, trip_distance, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, fare_amount, tip_amount, total_amount, buildingid_pickup, buildingid_dropoff, buildingtext_pickup, buildingtext_dropoff from old_nyc_taxi where (pickup_longitude between -180 and 180) and (pickup_latitude between -90 and 90) and (dropoff_longitude between -180 and 180) and  (dropoff_latitude between -90 and 90)",
            "visibility": "True"
        }
    ]
}
```

`response.json`:

```json
{
    "code":200,
    "message":"load data succeed!",
    "status":"success"
}
```

举个例子：

```shell
curl -X POST -H "Content-Type: application/json" -H "Authorization: Token YourToken" -d @./arctern/gui/server/db.json http://127.0.0.1:8080/load
```

### /dbs 获取数据库列表

method: GET
token: YES

```json
response:
    {
        "status": "success",
        "code": 200,
        "data": [
            {
                "id": "1",              //数据库id
                "name": "nyc taxi",     //数据库名字
                "type": "spark"         //数据库类型
            }
        ],
        "message": null
    }
```

### /db/tables 获取指定数据库的所有表名

method: POST
token: YES

```json
request:
    {
        "id": "1"                   //指定数据库id
    }
response:
    {
        "status": "success",
        "code": 200,
        "data": [
            "nyc_taxi"              //表名列表
        ],
        "message": null
    }
```

for example:

```shell
curl -X POST -H "Content-Type: application/json" -H "Authorization: Token yours" -d "{\"id\":\"1\"}" http://127.0.0.1:8080/db/tables
```

### /db/table/info 获取指定数据库表的详细信息

method: POST
token: YES

```json
request:
    {
        "id": "1",                              // 指定数据库id
        "table":  "nyc_taxi"        // 指定表名
    }
response:
    {
        "status": "success",
        "code": 200,
        "data": [   //表中的列
            {
                "col_name": "VendorID",
                "data_type": "string"
            }, {
                "col_name": "tpep_pickup_datetime",
                "data_type": "timestamp"
            }, {
                "col_name": "tpep_dropoff_datetime",
                "data_type": "timestamp"
            }, {
                "col_name": "passenger_count",
                "data_type": "bigint"
            }
        ],
        "message": null
    }
```

for example:

```shell
curl -X POST -H "Content-Type: application/json" -H "Authorization: Token yours" -d "{\"id\":\"1\",\"table\":\"nyc_taxi\"}" http://127.0.0.1:8080/db/table/info
```

### /db/query 查询某个数据库

method: POST
token: YES

```json
request:
    {
        "id": "1",              //指定数据库id
        "query":
        {
            // sql: 数据查询，result为[]，无params字段
            // point/heat/choropleth: 图片，result为base64图片数据，分别有params字段
            "type": "sql/point/heat/choropleth",
            "sql": ".....",     //需要执行的sql
            "params":
            {
                "width": 1900,      //图片宽度
                "height": 1410,     //图片高度

                //点图的附加参数
                "point" :
                {
                    "bounding_box": [-73.998427, 40.730309, -73.954348, 40.780816], //范围[x_min,y_min,x_max,y_max]
                    "coordinate_system": "EPSG:4326",                               //坐标系
                    "point_size": 3,                                                //点的大小
                    "point_color": "#2DEF4A",                                       //点的颜色
                    "opacity": 0.5                                                  //点的透明度
                },

                //权重图的附加参数
                "weighted" :
                {
                    "bounding_box": [-73.998427, 40.730309, -73.954348, 40.780816], //范围[x_min,y_min,x_max,y_max]
                    "coordinate_system": "EPSG:4326",                               //坐标系
                    "color_gradient": ["#0000FF", "#FF0000"],                                //颜色风格
                    "color_bound": [0, 2],                                          //颜色标尺
                    "size_bound": [0, 10],                                          //点大小标尺
                    "opacity": 1.0                                                  //透明度
                },

                //热力图的附加参数
                "heat":
                {
                    "bounding_box": [-73.998427, 40.730309, -73.954348, 40.780816], //范围[x_min,y_min,x_max,y_max]
                    "coordinate_system": "EPSG:4326",                               //坐标系
                    "map_zoom_level": 10                                            //缩放比
                },

                //轮廓图的附加擦数
                "choropleth":
                {
                    "bounding_box": [-73.984092, 40.753893, -73.977588, 40.756342], //范围
                    "coordinate_system": "EPSG:4326",                               //坐标系
                    "color_gradient": ["#0000FF", "#FF0000"],                                //颜色风格
                    "color_bound": [2.5, 5],                                        //标尺
                    "opacity" : 1                                                   //透明度
                }
            }
        }
    }
response:
    {
        "status": "success",
        "code": 200,
        "data":
        {
            "sql": ".....",     //执行的sql
            "err": false,
            "result": [             //sql结果列表
                row1, row2, row3    //row1,row2,row3分别为json对象
            ]
        },
        "message": null
    }
```

for example:

点图

```shell
curl --location --request POST 'http://localhost:8080/db/query' \
--header 'Authorization: Token YourToken' \
--header 'Content-Type: application/json' \
--data-raw '{
    "id": "1",
    "query": {
        "sql": "select ST_Point(pickup_longitude, pickup_latitude) as point from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude), ST_GeomFromText('\''POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'\''))",
        "type": "point",
        "params": {
            "width": 1024,
            "height": 896,
            "point": {
                "bounding_box": [
                    -75.37976,
                    40.191296,
                    -71.714099,
                    41.897445
                ],
                "coordinate_system": "EPSG:4326",
                "point_color": "#2DEF4A",
                "point_size": 3,
                "opacity": 0.5
            }
        }
    }
}'
```

其中json为

```json
{
    "id": "1",
    "query": {
        "sql": "select ST_Point(pickup_longitude, pickup_latitude) as point from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude), ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))",
        "type": "point",
        "params": {
            "width": 1024,
            "height": 896,
            "point": {
                "bounding_box": [
                    -75.37976,
                    40.191296,
                    -71.714099,
                    41.897445
                ],
                "coordinate_system": "EPSG:4326",
                "point_color": "#2DEF4A",
                "point_size": 3,
                "opacity": 0.5
            }
        }
    }
}
```

热力图

```shell
curl --location --request POST 'http://localhost:8080/db/query' \
--header 'Authorization: Token YourToken' \
--header 'Content-Type: application/json' \
--data-raw '{
    "id": "1",
    "query": {
        "sql": "select ST_Point(pickup_longitude, pickup_latitude) as point, passenger_count as w from nyc_taxi where ST_Within( ST_Point(pickup_longitude, pickup_latitude), ST_GeomFromText('\''POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'\''))",
        "type": "heat",
        "params": {
            "width": 1024,
            "height": 896,
            "heat": {
                "bounding_box": [
                    -75.37976,
                    40.191296,
                    -71.714099,
                    41.897445
                ],
                "coordinate_system": "EPSG:4326",
                "map_zoom_level": 10
            }
        }
    }
}'
```

其中json为

```json
{
    "id": "1",
    "query": {
        "sql": "select ST_Point(pickup_longitude, pickup_latitude) as point, passenger_count as w from nyc_taxi where ST_Within( ST_Point(pickup_longitude, pickup_latitude), ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))",
        "type": "heat",
        "params": {
            "width": 1024,
            "height": 896,
            "heat": {
                "bounding_box": [
                    -75.37976,
                    40.191296,
                    -71.714099,
                    41.897445
                ],
                "coordinate_system": "EPSG:4326",
                "map_zoom_level": 10
            }
        }
    }
}
```

轮廓图

```shell
curl --location --request POST 'http://localhost:8080/db/query' \
--header 'Authorization: Token YourToken' \
--header 'Content-Type: application/json' \
--data-raw '{
    "id": "1",
    "query": {
        "sql": "select ST_GeomFromText(buildingtext_dropoff) as wkt, passenger_count as w from nyc_taxi",
        "type": "choropleth",
        "params": {
            "width": 1024,
            "height": 896,
            "choropleth": {
                "bounding_box": [
                    -75.37976,
                    40.191296,
                    -71.714099,
                    41.897445
                ],
                "coordinate_system": "EPSG:4326",
                "color_gradient": ["#0000FF", "#FF0000"],
                "color_bound": [
                    2.5,
                    5
                ],
                "opacity": 1
            }
        }
    }
}'
```

其中json为

```json
{
    "id": "1",
    "query": {
        "sql": "select ST_GeomFromText(buildingtext_dropoff) as wkt, passenger_count as w from nyc_taxi",
        "type": "choropleth",
        "params": {
            "width": 1024,
            "height": 896,
            "choropleth": {
                "bounding_box": [
                    -75.37976,
                    40.191296,
                    -71.714099,
                    41.897445
                ],
                "coordinate_system": "EPSG:4326",
                "color_gradient": ["#0000FF", "#FF0000"],
                "color_bound": [2.5, 5],
                "opacity": 1
            }
        }
    }
}
```

权重图

```shell
curl --location --request POST 'http://localhost:8080/db/query' \
--header 'Authorization: Token eyJhbGciOiJIUzUxMiIsImlhdCI6MTU4NjI1NTE1OCwiZXhwIjoxNTg2ODU5OTU4fQ.eyJ1c2VyIjoiemlsbGl6In0.FbRX5ktderWVDtl1JzRN-yGZlWINAZiv3VUfe_WO-EZLDcxwvL2y2sBlm4lENpIFpNOU7lW1PhwtKhTRwGs85A' \
--header 'Content-Type: application/json' \
--data-raw '{
    "id": "1",
    "query": {
        "sql": "select ST_Point(pickup_longitude, pickup_latitude) as point, tip_amount as c, fare_amount as s from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude), ST_GeomFromText('\''POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'\''))",
        "type": "weighted",
        "params": {
            "width": 1024,
            "height": 896,
            "weighted": {
                "bounding_box": [
                    -75.37976,
                    40.191296,
                    -71.714099,
                    41.897445
                ],
                "color_gradient": ["#0000FF", "#FF0000"],
                "color_bound": [
                    0,
                    2
                ],
                "size_bound": [
                    0,
                    10
                ],
                "opacity": 1,
                "coordinate_system": "EPSG:4326"
            }
        }
    }
}'
```

其中json为：

```json
{
    "id": "1",
    "query": {
        "sql": "select ST_Point(pickup_longitude, pickup_latitude) as point, tip_amount as c, fare_amount as s from nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude), ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))",
        "type": "weighted",
        "params": {
            "width": 1024,
            "height": 896,
            "weighted": {
                "bounding_box": [
                    -75.37976,
                    40.191296,
                    -71.714099,
                    41.897445
                ],
                "color_gradient": ["#0000FF", "#FF0000"], 
                "color_bound": [
                    0,
                    2
                ],
                "size_bound": [
                    0,
                    10
                ],
                "opacity": 1,
                "coordinate_system": "EPSG:4326"
            }
        }
    }
}
```

## 代码测试

启动server：

```shell
cd arctern/gui/server
python manage.py -r 
```

运行restful api test：

```shell
cd arctern/gui/server/tests/restful
pytest --host=localhost --port=8080 --config=../../db.json
```

