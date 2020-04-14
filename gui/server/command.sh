curl --location --request GET 'http://192.168.2.29:9999/scope' \
--header 'Authorization: Token eyJhbGciOiJIUzUxMiIsImlhdCI6MTU4NjQ4NDM5NSwiZXhwIjoxNTg3MDg5MTk1fQ.eyJ1c2VyIjoiemlsbGl6In0.Od-3FaaiU6AkEqKaJiD3gNPCyHXyaNMIi2LkqrdpveeGZrgIH_4CNBrZ9dAQlyTYj9PFa6a1AD1vBGtViCVrvA' 

curl --location --request POST 'http://192.168.2.29:9999/scope' \
--header 'Authorization: Token eyJhbGciOiJIUzUxMiIsImlhdCI6MTU4NjQ4NDM5NSwiZXhwIjoxNTg3MDg5MTk1fQ.eyJ1c2VyIjoiemlsbGl6In0.Od-3FaaiU6AkEqKaJiD3gNPCyHXyaNMIi2LkqrdpveeGZrgIH_4CNBrZ9dAQlyTYj9PFa6a1AD1vBGtViCVrvA' \
--header 'Content-Type: application/json' \
--data-raw '{
	"scope_id":"scope1"
}'

curl --location --request DELETE 'http://192.168.2.29:9999/scope/scope5' \
--header 'Authorization: Token eyJhbGciOiJIUzUxMiIsImlhdCI6MTU4NjQ4NDM5NSwiZXhwIjoxNTg3MDg5MTk1fQ.eyJ1c2VyIjoiemlsbGl6In0.Od-3FaaiU6AkEqKaJiD3gNPCyHXyaNMIi2LkqrdpveeGZrgIH_4CNBrZ9dAQlyTYj9PFa6a1AD1vBGtViCVrvA'

curl --location --request POST 'http://192.168.2.29:9999/command' \
--header 'Authorization: Token eyJhbGciOiJIUzUxMiIsImlhdCI6MTU4NjQ4NDM5NSwiZXhwIjoxNTg3MDg5MTk1fQ.eyJ1c2VyIjoiemlsbGl6In0.Od-3FaaiU6AkEqKaJiD3gNPCyHXyaNMIi2LkqrdpveeGZrgIH_4CNBrZ9dAQlyTYj9PFa6a1AD1vBGtViCVrvA' \
--header 'Content-Type: application/json' \
--data-raw '{
	"scope_id":"scope1",
	"command":"import os\nimport json\n"
}'

curl --location --request POST 'http://192.168.2.29:9999/session' \
--header 'Authorization: Token eyJhbGciOiJIUzUxMiIsImlhdCI6MTU4NjQ4NDM5NSwiZXhwIjoxNTg3MDg5MTk1fQ.eyJ1c2VyIjoiemlsbGl6In0.Od-3FaaiU6AkEqKaJiD3gNPCyHXyaNMIi2LkqrdpveeGZrgIH_4CNBrZ9dAQlyTYj9PFa6a1AD1vBGtViCVrvA' \
--header 'Content-Type: application/json' \
--data-raw '{
    "scope_id": "arctern",
    "session_name": "db1",
    "app_name": "arctern",
    "master-addr": "local[*]",
    "configs":{
        "spark.executorEnv.GDAL_DATA": "",
        "spark.executorEnv.PROJ_LIB": ""
    },
    "envs": {
    }
}
'

curl --location --request POST 'http://192.168.2.29:9999/loadv2' \
--header 'Authorization: Token eyJhbGciOiJIUzUxMiIsImlhdCI6MTU4NjQ4NDM5NSwiZXhwIjoxNTg3MDg5MTk1fQ.eyJ1c2VyIjoiemlsbGl6In0.Od-3FaaiU6AkEqKaJiD3gNPCyHXyaNMIi2LkqrdpveeGZrgIH_4CNBrZ9dAQlyTYj9PFa6a1AD1vBGtViCVrvA' \
--header 'Content-Type: application/json' \
--data-raw '{
    "scope_id": "arctern",
    "session_name": "db1",
    "tables": [
        {
            "name": "old_nyc_taxi",
            "format": "csv",
            "path": "/arctern/gui/server/data/0_5M_nyc_taxi_and_building.csv",
            "options": {
                "header": "True",
                "delimiter": ","
            },
            "schema": [
                {
                    "VendorID": "string"
                },
                {
                    "tpep_pickup_datetime": "string"
                },
                {
                    "tpep_dropoff_datetime": "string"
                },
                {
                    "passenger_count": "long"
                },
                {
                    "trip_distance": "double"
                },
                {
                    "pickup_longitude": "double"
                },
                {
                    "pickup_latitude": "double"
                },
                {
                    "dropoff_longitude": "double"
                },
                {
                    "dropoff_latitude": "double"
                },
                {
                    "fare_amount": "double"
                },
                {
                    "tip_amount": "double"
                },
                {
                    "total_amount": "double"
                },
                {
                    "buildingid_pickup": "long"
                },
                {
                    "buildingid_dropoff": "long"
                },
                {
                    "buildingtext_pickup": "string"
                },
                {
                    "buildingtext_dropoff": "string"
                }
            ]
        },
        {
            "name": "nyc_taxi",
            "sql": "select VendorID, to_timestamp(tpep_pickup_datetime,'\''yyyy-MM-dd HH:mm:ss XXXXX'\'') as tpep_pickup_datetime, to_timestamp(tpep_dropoff_datetime,'\''yyyy-MM-dd HH:mm:ss XXXXX'\'') as tpep_dropoff_datetime, passenger_count, trip_distance, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, fare_amount, tip_amount, total_amount, buildingid_pickup, buildingid_dropoff, buildingtext_pickup, buildingtext_dropoff from old_nyc_taxi where (pickup_longitude between -180 and 180) and (pickup_latitude between -90 and 90) and (dropoff_longitude between -180 and 180) and  (dropoff_latitude between -90 and 90)"
        }
    ]
}'
