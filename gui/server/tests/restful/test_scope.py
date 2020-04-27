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

import pytest
import requests

original_table_name = "raw_data"
table_name = "nyctaxi"
csv_path = "/arctern/gui/server/data/0_5M_nyc_taxi_and_building.csv"
SCOPE = "nyc_taxi"

def _get_line_count(file):
    with open(file, "r") as f:
        return len(f.readlines())

class TestScope():
    @pytest.mark.run(order=1)
    def test_create_scope(self, host, port):
        url = "http://" + host + ":" + port + "/scope"
        r = requests.post(url=url)
        print(r.text)
        assert r.status_code == 200
        global SCOPE # pylint: disable=global-statement
        SCOPE = r.json()['scope']

    @pytest.mark.run(order=2)
    def test_load_file(self, host, port):
        url = "http://" + host + ":" + port + "/loadfile"
        payload = {
            "scope": SCOPE,
            "tables": [
                    {
                        "name": original_table_name,
                        "format": "csv",
                        "path": csv_path,
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
                    ]
                }
            ]
        }
        r = requests.post(url=url, json=payload)
        print(r.text)
        assert r.status_code == 200

    # TODO: neccessary for /savetable? not convenient for cleaning up

    @pytest.mark.run(order=3)
    def test_table_schema(self, host, port):
        url = "http://" + host + ":" + port + "/table/schema?table={}&scope={}".format(original_table_name, SCOPE)
        r = requests.get(url=url)
        print(r.text)
        assert r.status_code == 200
        assert len(r.json()['schema']) == 16

    @pytest.mark.run(order=4)
    def test_num_rows(self, host, port):
        url = "http://" + host + ":" + port + "/query"
        sql = "select count(*) as num_rows from {}".format(original_table_name)
        payload = {
            "scope": SCOPE,
            "sql": sql,
            "collect_result": "1"
        }
        r = requests.post(url=url, json=payload)
        print(r.text)
        assert r.status_code == 200
        assert len(r.json()['result']) == 1
        assert r.json()['result'][0]['num_rows'] == _get_line_count(csv_path) - 1

    @pytest.mark.run(order=5)
    def test_query(self, host, port):
        url = "http://" + host + ":" + port + "/query"
        limit = 1
        sql = "select * from {} limit {}".format(original_table_name, limit)
        payload = {
            "scope": SCOPE,
            "sql": sql,
            "collect_result": "1"
        }
        r = requests.post(url=url, json=payload)
        print(r.text)
        assert r.status_code == 200
        assert len(r.json()['result']) == limit

    @pytest.mark.run(order=6)
    def test_create_table(self, host, port):
        url = "http://" + host + ":" + port + "/query"
        payload = {
            "scope": SCOPE,
            "sql": "create table {} as (select VendorID, to_timestamp(tpep_pickup_datetime,'yyyy-MM-dd HH:mm:ss XXXXX') as tpep_pickup_datetime, to_timestamp(tpep_dropoff_datetime,'yyyy-MM-dd HH:mm:ss XXXXX') as tpep_dropoff_datetime, passenger_count, trip_distance, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, fare_amount, tip_amount, total_amount, buildingid_pickup, buildingid_dropoff, buildingtext_pickup, buildingtext_dropoff from {} where (pickup_longitude between -180 and 180) and (pickup_latitude between -90 and 90) and (dropoff_longitude between -180 and 180) and  (dropoff_latitude between -90 and 90))".format(table_name, original_table_name),
            "collect_result": "0"
        }
        r = requests.post(url=url, json=payload)
        print(r.text)
        assert r.status_code == 200

    @pytest.mark.run(order=7)
    def test_pointmap(self, host, port):
        url = "http://" + host + ":" + port + "/pointmap"
        payload = {
            "scope": SCOPE,
            "sql": "select ST_Point(pickup_longitude, pickup_latitude) as point from {} where ST_Within(ST_Point(pickup_longitude, pickup_latitude), ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))".format(table_name),
            "params": {
                "width": 1024,
                "height": 896,
                "bounding_box": [-80.37976, 35.191296, -70.714099, 45.897445],
                "coordinate_system": "EPSG:4326",
                "point_color": "#2DEF4A",
                "point_size": 3,
                "opacity": 0.5
            }
        }
        r = requests.post(url=url, json=payload)
        assert r.status_code == 200
        print(r.text)
        # assert r.json()["result"] is not None

    @pytest.mark.run(order=8)
    def test_weighted_pointmap(self, host, port):
        url = "http://" + host + ":" + port + "/weighted_pointmap"
        payload = {
            "scope": SCOPE,
            "sql": "select ST_Point(pickup_longitude, pickup_latitude) as point, tip_amount as c, fare_amount as s from {} where ST_Within(ST_Point(pickup_longitude, pickup_latitude), ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))".format(table_name),
            "params": {
                "width": 1024,
                "height": 896,
                "bounding_box": [-80.37976, 35.191296, -70.714099, 45.897445],
                "color_gradient": ["#0000FF", "#FF0000"],
                "color_bound": [0, 2],
                "size_bound": [0, 10],
                "opacity": 1.0,
                "coordinate_system": "EPSG:4326"
            }
        }
        r = requests.post(url=url, json=payload)
        assert r.status_code == 200
        print(r.text)
        # assert r.json()["result"] is not None

    @pytest.mark.run(order=9)
    def test_heatmap(self, host, port):
        url = "http://" + host + ":" + port + "/heatmap"
        payload = {
            "scope": SCOPE,
            "sql": "select ST_Point(pickup_longitude, pickup_latitude) as point, passenger_count as w from {} where ST_Within(ST_Point(pickup_longitude, pickup_latitude), ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))".format(table_name),
            "params": {
                "width": 1024,
                "height": 896,
                "bounding_box": [-80.37976, 35.191296, -70.714099, 45.897445],
                "coordinate_system": "EPSG:4326",
                "map_zoom_level": 10,
                "aggregation_type": "sum"
            }
        }
        r = requests.post(url=url, json=payload)
        assert r.status_code == 200
        print(r.text)
        # assert r.json()["result"] is not None

    @pytest.mark.run(order=10)
    def test_choroplethmap(self, host, port):
        url = "http://" + host + ":" + port + "/choroplethmap"
        payload = {
            "scope": SCOPE,
            "sql": "select ST_GeomFromText(buildingtext_dropoff) as wkt, passenger_count as w from {} where (buildingtext_dropoff!='')".format(table_name),
            "params": {
                "width": 1024,
                "height": 896,
                "bounding_box": [-80.37976, 35.191296, -70.714099, 45.897445],
                "coordinate_system": "EPSG:4326",
                "color_gradient": ["#0000FF", "#FF0000"],
                "color_bound": [2.5, 5],
                "opacity": 1,
                "aggregation_type": "sum"
            }
        }
        r = requests.post(url=url, json=payload)
        assert r.status_code == 200
        print(r.text)
        # assert r.json()["result"] is not None

    @pytest.mark.run(order=11)
    def test_icon_viz(self, host, port):
        url = "http://" + host + ":" + port + "/icon_viz"
        import os
        dir_path = os.path.dirname(os.path.realpath(__file__))
        png_path = dir_path + "/taxi.png"
        payload = {
            "scope": SCOPE,
            "sql": "select ST_Point(pickup_longitude, pickup_latitude) as point from {} where ST_Within(ST_Point(pickup_longitude, pickup_latitude), ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))".format(table_name),
            "params": {
                'width': 1024,
                'height': 896,
                'bounding_box': [-75.37976, 40.191296, -71.714099, 41.897445],
                'coordinate_system': 'EPSG:4326',
                'icon_path': png_path
            }
        }
        r = requests.post(url=url, json=payload)
        assert r.status_code == 200
        print(r.text)
        # assert r.json()["result"] is not None

    @pytest.mark.run(order=12)
    def test_drop_table(self, host, port):
        url = "http://" + host + ":" + port + '/query'
        sql1 = "drop table if exists {}".format(table_name)
        sql2 = "drop table if exists {}".format(original_table_name)
        payload1 = {
            "scope": SCOPE,
            "sql": sql1,
            "collect_result": "0"
        }
        payload2 = {
            "scope": SCOPE,
            "sql": sql2,
            "collect_result": "0"
        }
        r = requests.post(url=url, json=payload1)
        print(r.text)
        assert r.status_code == 200
        r = requests.post(url=url, json=payload2)
        print(r.text)
        assert r.status_code == 200

    @pytest.mark.run(order=13)
    def test_command(self, host, port):
        url = "http://" + host + ":" + port + '/command'
        command = """
from __future__ import print_function

import sys
from random import random
from operator import add

partitions = 2
n = 100000 * partitions

def f(_):
    x = random() * 2 - 1
    y = random() * 2 - 1
    return 1 if x ** 2 + y ** 2 <= 1 else 0

count = spark.sparkContext.parallelize(range(1, n + 1), partitions).map(f).reduce(add)
print("Pi is roughly %f" % (4.0 * count / n))
        """
        payload = {
            "scope": SCOPE,
            "command": command
        }
        r = requests.post(url=url, json=payload)
        print(r.text)
        assert r.status_code == 200

    @pytest.mark.run(order=14)
    def test_remove_scope(self, host, port):
        scope = SCOPE
        url = "http://" + host + ":" + port + "/scope/" + scope
        r = requests.delete(url=url)
        print(r.text)
        assert r.status_code == 200
