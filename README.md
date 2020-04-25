<img src="./doc/img/icon/arctern-color.png" width = "200">

[中文README](./README_CN.md)

> Notice: Arctern is still in development and the 0.1.0 version is expected to be released in April 2020.

## Overview

Arctern is a geospatial analytics engine for massive-scale data. Compared with other geospatial analytics tools, Arctern aims at providing the following advantages:

1. Provides domain-specific APIs to improve the development efficiency of upper-level applications.
2. Provides extensible, low-cost distributed solutions.
3. Provides GPU acceleration for geospatial analytics algorithms.
4. Support hybrid analysis with GIS, SQL, and ML functionalities.

## Architecture

The following figure shows the architecture of Arctern 0.1.0.  

<img src="./doc/img/v0.1.0_intro/arctern_arch_v0.1.0.png" width = "700">

Arctern includes two components: GIS and Visualization. Arctern 0.1.0 supports most frequently used 35 GIS APIs in the OGC standard, including construction, access, correlation analysis, measurement for geometric objects. The visualization component is responsible for rendering geometry objects. It provides standard Vega rendering APIs. Different from traditional web rendering, Arctern uses server-side rendering and can render choropleths, heatmaps, and scatter plots for massive-scale data.  With a set of unified APIs, Arctern provides both CPU and GPU based implementations for geospatial data analytics and visualization.

For data format, Arctern supports standard numeric types, WKB formats, and files with JSON, CSV, and parquet format. Arctern organizes data in the memory in a column-based manner according to the Arrow standard. In this way, Arctern supports zero-copy data exchange with external systems.

Arctern includes three types of column-based interface: C++ API, Python API, and Spark API. The C++ APIs pass arguments in Arrow format, Python and Spark APIs pass arguments in dataframe format. Because Spark will start to support GPU resource management since the 3.0 version, the Spark interface of Arctern only supports Spark 3.0.

## Code example 

```python
# Invoke Arctern API in PySpark

from pyspark.sql import SparkSession
from arctern_pyspark import register_funcs, heatmap
from arctern.util import save_png
from arctern.util.vega import vega_heatmap 

if __name__== "__main__":
    spark = SparkSession \
            .builder \
            .appName("Arctern-PySpark example") \
            .getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    register_funcs(spark)

    df = spark.read.format("csv") \
         .option("header", True) \ 
         .option("delimiter", ",") \
         .schema("passenger_count long,  pickup_longitude double, pickup_latitude double") \
         .load("file:///tmp/0_5M_nyc_taxi_and_building.csv") \
         .cache()
    df.createOrReplaceTempView("nyc_taxi")
        
    res = spark.sql(
        "select ST_Point(pickup_longitude, pickup_latitude) as point, passenger_count as w \
        from nyc_taxi \
        where ST_Within(ST_Point(pickup_longitude, pickup_latitude), 'POLYGON ((-73.998427 40.730309, \
                                                                                -73.954348 40.730309, \
                                                                                -73.954348 40.780816, \
                                                                                -73.998427 40.780816, \
                                                                                -73.998427 40.730309))')")

    vega = vega_heatmap(1024, 896, 10.0, [-73.998427, 40.730309, -73.954348, 40.780816], 'EPSG:4326')
    res = heatmap(res, vega)
    save_png(res, '/tmp/heatmap.png')

    spark.catalog.dropTempView("nyc_taxi")

    spark.stop()
```

## Visualization

Arctern will be open sourced along with Sulidae, which is a front-end visualization system developed by ZILLIZ and provides hybrid visualization solutions with both web frontend and server-side rendering. Sulidae combines the speed and flexibility of web frontend rendering and massive-scale data rendering capability of the backend.

Arctern 0.1.0 is compatible with Sulidae. The following figures show the visualization effects of a headmap and a choropleth with 10 million data.

<img src="./doc/img/v0.1.0_intro/heat_map.png" width = "700">

<img src="./doc/img/v0.1.0_intro/contour_map.png" width = "700">

## Arctern roadmap

### v0.1.0

1. Support most frequently used 35 GIS APIs in the OGC standard.
2. Support rendering choropleths, heatmaps, and scatter plots for massive-scale datasets.
3. Provide C++, Python, and Spark APIs that comply with Arrow standard.
4. Arctern engine with CPU based implementation.
5. Arctern engine with GPU based implementation.
6. Compatibility with Sulidae.
7. Documentation for installation, deployment, and API reference.

### v0.2.0

1. Domain-specific API for trace analysis and geospatial data analysis.
2. Geospatial indexes for domain-specific API.
3. Performance optimization in Spark 3.0.
4. Support more GIS APIs.
5. Continuously improve system stability.

### In progress:

#### Completed by 2020.03.10

1. Support most frequently used 35 GIS APIs in the OGC standard.
2. Support rendering horopleths, heatmaps, and scatter plots for massive-scale datasets.
3. Support C++, Python, and Spark APIs that comply with Arrow standard.
4. Arctern engine with CPU support.
5. Arctern engine with GPU support.

### Contact us

#### Email

support@zilliz.com

##### ZILLIZ Wechat

<img src="./doc/img/v0.1.0_intro/zilliz.png" width = "200" align=left>
