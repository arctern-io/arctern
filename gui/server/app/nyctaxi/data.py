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

from app.common import spark

GLOBAL_TABLE_LIST = []


def init():
    """
    load nyc taxi data to spark memory
    """
    print('nyctaxi.data.init')

    import os
    dirpath = os.path.split(os.path.realpath(__file__))[0]
    csvpath = dirpath + '/../../data/0_5M_nyc_taxi_and_building.csv'
    old_nyctaix_df = spark.INSTANCE.session.read.format("csv") \
        .option("header", True) \
        .option("delimiter", ",") \
        .schema("VendorID string, \
            tpep_pickup_datetime string, \
            tpep_dropoff_datetime string, \
            passenger_count long, \
            trip_distance double, \
            pickup_longitude double, \
            pickup_latitude double, \
            dropoff_longitude double, \
            dropoff_latitude double, \
            fare_amount double, \
            tip_amount double, \
            total_amount double, \
            buildingid_pickup long, \
            buildingid_dropoff long, \
            buildingtext_pickup string, \
            buildingtext_dropoff string") \
        .load(csvpath)
    old_nyctaix_df.createOrReplaceGlobalTempView("old_nyc_taxi")

    nyctaix_df = spark.INSTANCE.session.sql("select VendorID, \
            to_timestamp(tpep_pickup_datetime,'yyyy-MM-dd HH:mm:ss XXXXX') as tpep_pickup_datetime, \
            to_timestamp(tpep_dropoff_datetime,'yyyy-MM-dd HH:mm:ss XXXXX') as tpep_dropoff_datetime, \
                passenger_count, \
                trip_distance, \
                pickup_longitude, \
                pickup_latitude, \
                dropoff_longitude, \
                dropoff_latitude, \
                fare_amount, \
                tip_amount, \
                total_amount, \
                buildingid_pickup, \
                buildingid_dropoff, \
                buildingtext_pickup, \
                buildingtext_dropoff \
            from global_temp.old_nyc_taxi \
            where   (pickup_longitude between -180 and 180) and \
                    (pickup_latitude between -90 and 90) and \
                    (dropoff_longitude between -180 and 180) and \
                    (dropoff_latitude between -90 and 90)  ") \
        .cache()
    nyctaix_df.createOrReplaceGlobalTempView("nyc_taxi")

    GLOBAL_TABLE_LIST.append("global_temp.nyc_taxi")
