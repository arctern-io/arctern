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

import json

def test_dbs(client, headers):
    response = client.get(
        '/dbs',
        headers=headers
    )
    assert response.status_code == 200
    assert response.json['data'][0]['id'] == '1'
    assert response.json['data'][0]['name'] == 'nyc taxi'
    assert response.json['data'][0]['type'] == 'spark'

def test_tables(client, headers):
    # case 1: no id keyword in request.json
    response = client.post(
        '/db/tables',
        headers=headers
    )
    assert response.json['code'] == - 1
    assert response.json['message'] == 'json error: id is not exist'
    assert response.json['status'] == 'error'

    # case 2: invalid keyword in request.json
    response = client.post(
        '/db/tables',
        data=json.dumps(dict(invalidarg=3)),
        content_type='application/json',
        headers=headers
    )
    assert response.json['code'] == - 1
    assert response.json['message'] == 'json error: id is not exist'
    assert response.json['status'] == 'error'

    # case 3: corrent query format
    response = client.post(
        '/db/tables',
        data=json.dumps(dict(id=1)),
        content_type='application/json',
        headers=headers
    )
    assert response.status_code == 200
    assert response.json['data'][0] == 'global_temp.nyc_taxi'

    # TODO: check nonexistent id

def test_table_info(client, headers):
    # case 1: no id and table keyword in request.json
    response = client.post(
        '/db/table/info',
        headers=headers
    )
    assert response.json['status'] == 'error'
    assert response.json['code'] == -1
    assert response.json['message'] == 'query format error'

    # case 2: corrent query format
    response = client.post(
        '/db/table/info',
        data=json.dumps(dict(id=1, table='global_temp.nyc_taxi')),
        content_type='application/json',
        headers=headers
    )
    assert response.status_code == 200
    # TODO: check data field in response.json

    # TODO: check nonexistent id or table

def test_query(client, headers):
    # case 1: pointmap
    pointmap_request_dict = {
        'id': '1',
        'query': {
            'sql': '''
            select ST_Point(pickup_longitude, pickup_latitude) as point
            from global_temp.nyc_taxi
            where ST_Within(
                ST_Point(pickup_longitude, pickup_latitude),
                "POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))"
                )
            ''',
            'type': 'point',
            'params': {
                'width': 1024,
                'height': 896,
                'point': {
                    'bounding_box': [-73.998427, 40.730309, -73.954348, 40.780816],
                    'coordinate': 'EPSG:4326',
                    'stroke_width': 3,
                    'stroke': '#2DEF4A',
                    'opacity': 0.5
                }
            }
        }
    }
    response = client.post(
        '/db/query',
        data=json.dumps(pointmap_request_dict),
        content_type='application/json',
        headers=headers
    )
    assert response.status_code == 200

    # case 2: heatmap
    heatmap_request_dict = {
        'id': '1',
        'query': {
            'sql': '''
            select ST_Point(pickup_longitude, pickup_latitude) as point, passenger_count as w
            from global_temp.nyc_taxi
            where ST_Within(
                ST_Point(pickup_longitude, pickup_latitude),
                'POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'
                )
            ''',
            'type': 'heat',
            'params': {
                'width': 1024,
                'height': 896,
                'heat': {
                    'bounding_box': [-73.998427, 40.730309, -73.954348, 40.780816],
                    'coordinate': 'EPSG:4326',
                    'map_scale': 10
                }
            }
        }
    }
    response = client.post(
        '/db/query',
        data=json.dumps(heatmap_request_dict),
        content_type='application/json',
        headers=headers
    )
    assert response.status_code == 200

    # case 3: choropleth map
    choropleth_map_request_dict = {
        'id': '1',
        'query': {
            'sql': '''
            select buildingtext_dropoff as wkt, passenger_count as w
            from global_temp.nyc_taxi
            ''',
            'type': 'choropleth',
            'params': {
                'width': 1024,
                'height': 896,
                'choropleth': {
                    'bounding_box': [-73.998427, 40.730309, -73.954348, 40.780816],
                    'coordinate': 'EPSG:4326',
                    'color_style': 'blue_to_red',
                    'rule': [2.5, 5],
                    'opacity': 1
                }
            }
        }
    }
    response = client.post(
        '/db/query',
        data=json.dumps(choropleth_map_request_dict),
        content_type='application/json',
        headers=headers
    )
    assert response.status_code == 200
