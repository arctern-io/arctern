# arctern-gui-server

一个简易的RESTFUL接口的web服务器，采用python语言编写，主要是为了使用spark服务。

## 代码结构

## 配置文件

conf/config.init为配置文件，可以根据情况自行修改，其中：

```bash
[http]
port = 8080          # http服务器的监听端口

[spark]
# lcoal[*]  local mode
# yarn      hadoop/yarn mode, need env YARN_CONF_DIR and HADOOP_CONF_DIR
master-addr = local[*]
# python path for executor
executor-python = /home/gxz/miniconda3/envs/arctern/bin/python

```

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

在`https://github.com/zilliztech/arctern-tutorial/tree/master/data`下，获取0_5M_nyc_taxi_and_building.csv，
保持文件不变，放在data目录下

## 启动web服务

运行一下命令可以启动web服务

```bash
python manage.py -r
```

## 与spark服务对接

### spark local mode

需要设置 spark.executorEnv.PYSPARK_PYTHON
当前该字段值配置在： config.ini 中的 [spark] executor-python

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
            "global_temp.nyc_taxi"              //表名列表
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
        "table":  "global_temp.nyc_taxi"        // 指定表名
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
curl -X POST -H "Content-Type: application/json" -H "Authorization: Token yours" -d "{\"id\":\"1\",\"table\":\"global_temp.nyc_taxi\"}" http://127.0.0.1:8080/db/table/info
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
                    "coordinate": "EPSG:4326",                                      //坐标系
                    "stroke_width": 3,      //点的大小
                    "stroke": "#2DEF4A",    //点的颜色
                    "opacity": 0.5          //点的透明度
                },

                //热力图的附加参数
                "heat":
                {
                    "bounding_box": [-73.998427, 40.730309, -73.954348, 40.780816], //范围[x_min,y_min,x_max,y_max]
                    "coordinate": "EPSG:4326",                                      //坐标系
                    "map_scale": 10         //缩放比
                },

                //轮廓图的附加擦数
                "choropleth":
                {
                    "bounding_box": [-73.984092, 40.753893, -73.977588, 40.756342], //范围
                    "coordinate": "EPSG:4326",                                      //坐标系
                    "color_style": "blue_to_red",                                   //颜色风格
                    "rule": [2.5, 5],                                               //标尺
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
curl -X POST -H "Content-Type: application/json" -H "Authorization: Token yours" -d "{\"id\":\"1\",\"query\":{\"sql\":\"select ST_Point(pickup_longitude, pickup_latitude) as point from global_temp.nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude), 'POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))')\",\"type\":\"point\",\"params\":{\"width\":1024,\"height\":896,\"point\":{\"bounding_box\":[-73.998427,40.730309,-73.954348,40.780816],\"coordinate\":\"EPSG:4326\",\"stroke_width\":3,\"stroke\":\"#2DEF4A\",\"opacity\":0.5}}}}" http://127.0.0.1:8080/db/query
```

其中json为

```json
{
    "id": "1",
    "query": {
    "sql": "select ST_Point(pickup_longitude, pickup_latitude) as point from global_temp.nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude), 'POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))')",
        "type": "point",
        "params": {
            "width": 1024,
            "height": 896,
            "point": {
                "bounding_box": [-73.998427, 40.730309, -73.954348, 40.780816],
                "coordinate": "EPSG:4326",
                "stroke_width": 3,
                "stroke": "#2DEF4A",
                "opacity": 0.5
            }
        }
    }
}
```

热力图

```shell
curl -X POST -H "Content-Type: application/json" -H "Authorization: Token yours" -d "{\"id\":\"1\",\"query\":{\"sql\":\"select ST_Point(pickup_longitude, pickup_latitude) as point, passenger_count as w from global_temp.nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude),  'POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))')\",\"type\":\"heat\",\"params\":{\"width\":1024,\"height\":896,\"heat\":{\"bounding_box\":[-73.998427,40.730309,-73.954348,40.780816],\"coordinate\":\"EPSG:4326\",\"map_scale\":10}}}}" http://127.0.0.1:8080/db/query
```

其中json为

```json
{
    "id": "1",
    "query": {
        "sql": "select ST_Point(pickup_longitude, pickup_latitude) as point, passenger_count as w from global_temp.nyc_taxi where ST_Within(ST_Point(pickup_longitude, pickup_latitude),  'POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))')",
        "type": "heat",
        "params": {
            "width": 1024,
            "height": 896,
            "heat": {
                "bounding_box": [-73.998427, 40.730309, -73.954348, 40.780816],
                "coordinate": "EPSG:4326",
                "map_scale": 10
            }
        }
    }
}
```

轮廓图

```shell
curl -X POST -H "Content-Type: application/json" -H "Authorization: Token yours" -d "{\"id\":\"1\",\"query\":{\"sql\":\"select buildingtext_dropoff as wkt, passenger_count as w from global_temp.nyc_taxi\",\"type\":\"heat\",\"params\":{\"width\":1024,\"height\":896,\"choropleth\":{\"bounding_box\":[-73.984092,40.753893,-73.977588,40.756342],\"coordinate\":\"EPSG:4326\",\"color_style\":\"blue_to_red\",\"rule\":[2.5,5],\"opacity\":1}}}}" http://127.0.0.1:8080/db/query
```

其中json为

```json
{
    "id": "1",
    "query": {
        "sql": "select buildingtext_dropoff as wkt, passenger_count as w from global_temp.nyc_taxi",
        "type": "choropleth",
        "params": {
            "width": 1024,
            "height": 896,
            "choropleth": {
                "bounding_box": [-73.984092, 40.753893, -73.977588, 40.756342],
                "coordinate": "EPSG:4326",
                "color_style": "blue_to_red",
                "rule": [2.5, 5],
                "opacity": 1
            }
        }
    }
}
```
