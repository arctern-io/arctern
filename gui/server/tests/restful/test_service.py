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

# pylint: disable=redefined-outer-name

import pytest
import requests

@pytest.fixture(scope='function')
def dbid(host, port, headers):
    url = 'http://' + host + ':' + port + '/dbs'
    response = requests.get(
         url=url,
         headers=headers,
    )
    return response.json()['data'][0]['id']

@pytest.fixture(scope='function')
def table_name(host, port, headers, dbid):
    url = 'http://' + host + ':' + port + '/db/tables'
    response = requests.post(
        url=url,
        json={'id': dbid},
        headers=headers,
    )
    return response.json()['data'][0]

def test_dbs(host, port, headers):
    url = 'http://' + host + ':' + port + '/dbs'
    response = requests.get(
         url=url,
         headers=headers,
    )
    assert response.status_code == 200

def test_tables(host, port, headers, dbid):
    url = 'http://' + host + ':' + port + '/db/tables'
    # case 1: no id keyword in request.json
    response = requests.post(
         url=url,
         headers=headers,
    )
    assert response.json()['code'] == - 1
    assert response.json()['message'] == 'json error: id is not exist'
    assert response.json()['status'] == 'error'

    # case 2: invalid keyword in request.json
    response = requests.post(
         url=url,
         json={'invalidarg': 3},
         headers=headers,
    )
    assert response.json()['code'] == - 1
    assert response.json()['message'] == 'json error: id is not exist'
    assert response.json()['status'] == 'error'

    # case 3: correct query format
    response = requests.post(
         url=url,
         json={'id': dbid},
         headers=headers,
    )
    assert response.status_code == 200

    # TODO: check nonexistent id

def test_table_info(host, port, headers, dbid, table_name):
    url = 'http://' + host + ':' + port + '/db/table/info'
    # case 1: no id and table keyword in request.json
    response = requests.post(
         url=url,
         headers=headers,
    )
    assert response.json()['status'] == 'error'
    assert response.json()['code'] == -1
    assert response.json()['message'] == 'query format error'

    # case 2: correct query format
    response = requests.post(
         url=url,
         json={'id': dbid, 'table': table_name},
         headers=headers,
    )
    assert response.status_code == 200

def test_query(host, port, headers, dbid, table_name):
    url = 'http://' + host + ':' + port + '/db/query'
    # case 1: pointmap
    pointmap_request_dict = {
        'id': dbid,
        'query': {
             'sql': '''
                    select ST_Point(pickup_longitude, pickup_latitude) as point
                    from {}
                    where ST_Within(
                        ST_Point(pickup_longitude, pickup_latitude),
                        ST_GeomFromText("POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))")
                        )
             '''.format(table_name),
             'type': 'point',
             'params': {
                  'width': 1024,
                  'height': 896,
                  'point': {
                    'bounding_box': [-75.37976, 40.191296, -71.714099, 41.897445],
                       'coordinate_system': 'EPSG:4326',
                       'point_color': '#2DEF4A',
                       'point_size': 3,
                       'opacity': 0.5
                  }
             }
        }
    }
    response = requests.post(
         url=url,
         json=pointmap_request_dict,
         headers=headers,
    )
    assert response.status_code == 200
    assert response.json()['data']['result'] is not None

    # case 2: heatmap
    heatmap_request_dict = {
        'id': dbid,
        'query': {
            'sql': '''
            select ST_Point(pickup_longitude, pickup_latitude) as point, passenger_count as w
            from {}
            where ST_Within(
                ST_Point(pickup_longitude, pickup_latitude),
                ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))')
                )
            '''.format(table_name),
            'type': 'heat',
            'params': {
                'width': 1024,
                'height': 896,
                'heat': {
                    'bounding_box': [-75.37976, 40.191296, -71.714099, 41.897445],
                    'coordinate_system': 'EPSG:4326',
                    'map_zoom_level': 10,
                    'aggregation_type': 'sum'
                }
            }
        }
    }
    response = requests.post(
         url=url,
         json=heatmap_request_dict,
         headers=headers,
    )
    assert response.status_code == 200
    assert response.json()['data']['result'] is not None

    # case 3: choropleth map
    choropleth_map_request_dict = {
        'id': dbid,
        'query': {
            'sql': '''
            select ST_GeomFromText(buildingtext_dropoff) as wkt, passenger_count as w
            from {} where (buildingtext_dropoff!='')
            '''.format(table_name),
            'type': 'choropleth',
            'params': {
                'width': 1024,
                'height': 896,
                'choropleth': {
                    'bounding_box': [-75.37976, 40.191296, -71.714099, 41.897445],
                    'coordinate_system': 'EPSG:4326',
                    'color_gradient': ["#0000FF", "#FF0000"],
                    'color_bound': [2.5, 5],
                    'opacity': 1,
                    'aggregation_type': 'sum'
                }
            }
        }
    }
    response = requests.post(
         url=url,
         json=choropleth_map_request_dict,
         headers=headers,
    )
    assert response.status_code == 200
    assert response.json()['data']['result'] is not None

    # case 4: weighted pointmap
    weighted_pointmap_request_dict = {
        'id': dbid,
        'query': {
            'sql': '''
            select ST_Point(pickup_longitude, pickup_latitude) as point, tip_amount as c, fare_amount as s
            from {}
            where ST_Within(
                ST_Point(pickup_longitude, pickup_latitude),
                ST_GeomFromText('POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))'))
            '''.format(table_name),
            'type': 'weighted',
            'params': {
                'width': 1024,
                'height': 896,
                'weighted': {
                    'bounding_box': [-75.37976, 40.191296, -71.714099, 41.897445],
                    'color_gradient': ["#0000FF", "#FF0000"],
                    'color_bound': [0, 2],
                    'size_bound': [0, 10],
                    'opacity': 1.0,
                    'coordinate_system': 'EPSG:4326'
                }
            }
        }
    }
    response = requests.post(
        url=url,
        json=weighted_pointmap_request_dict,
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()['data']['result'] is not None

    # case 5: icon_viz
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    png_path = dir_path + "/taxi.png"
    icon_viz_request_dict = {
        'id': dbid,
        'query': {
            'sql': '''
                    select ST_Point(pickup_longitude, pickup_latitude) as point
                    from {}
                    where ST_Within(
                        ST_Point(pickup_longitude, pickup_latitude),
                        ST_GeomFromText("POLYGON ((-73.998427 40.730309, -73.954348 40.730309, -73.954348 40.780816 ,-73.998427 40.780816, -73.998427 40.730309))")
                        )
             '''.format(table_name),
            'type': 'icon',
            'params': {
                'width': 1024,
                'height': 896,
                'icon': {
                    'bounding_box': [-75.37976, 40.191296, -71.714099, 41.897445],
                    'coordinate_system': 'EPSG:4326',
                    'icon_path': png_path
                }
            }
        }
    }
    response = requests.post(
        url=url,
        json=icon_viz_request_dict,
        headers=headers,
    )
    assert response.status_code == 200
    assert response.json()['data']['result'] is not None
