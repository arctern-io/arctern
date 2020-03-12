<img src="./doc/img/icon/arctern-color.png" width = "200">

[中文README](./README_CN.md)

> Notice: Arctern is still in development and the 0.1.0 version is expected to be released in April 2020.

## Overview

Arctern is a geospatial analytics engine for massive-scale data. Compared with other geospatial analytics tools, Arctern has the following advantages:

1. Provides domain-specific APIs to improve the development efficiency of upper-level applications.
2. Provides extensible, low-cost distributed solutions.
3. Provides GPU acceleration for geospatial analytics algorithms.
4. Provides hybrid analysis with GIS, SQL, and ML.

## Architecture

The following figure shows the architecture of Arctern 0.1.0.  

<img src="./doc/img/v0.1.0_intro/arctern_arch_v0.1.0.png" width = "700">

Arctern includes two components: GIS and Visualization. Arctern 0.1.0 includes most frequently used 35 GIS APIs in the OGC standard, including construction, access, correlation analysis, measurement for geometric objects. The visualization component is responsible for rendering geometry objects and provides APIs according to the Vega standard. Different from traditional web rendering, Arctern uses server-side rendering and can render choropleths, heatmaps, and scatter plots for billion-scale data. In 0.1.0, geospatial data analytics and visualization with both CPU and GPU based implementation. Arctern provides a unified set of APIs for you to determine whether to use GPU acceleration based on your own needs.

For data interfaces, Arctern supports standard numeric types, WKB formats, and files with JSON, CSV, and parquet format. Arctern organizes data in the memory in a column-based data manner according to the Arrow standard. In this way, Arctern supports zero-copy data exchange with external systems.

For invocation interfaces, Arctern includes three column-based interfaces: C++ API, Python API, and Spark API. The APIs transfer and return arguments with the Arrow standard. Because Spark will start to support GPU resource management since the 3.0 version, the Spark interface of Arctern only supports Spark 3.0.

## Code sample

```python
# Invoke Arctern API in PySpark

from pyspark.sql import SparkSession
import arctern

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Arctern-PySpark example") \
        .getOrCreate()

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    arctern.pyspark.register(spark)

    within_df = spark.read.json('./example.json').cache()
    within_df.createOrReplaceTempView("within")
    spark.sql("select ST_Within_UDF(geo0, geo1) from within").show()
    spark.stop()
```

## Visualization

Arctern will be open sourced in sync with Sulidae, which is a front-end visualization system developed by ZILLIZ and provides hybrid visualization solutions with both web frontend and server-side rendering. Sulidae combines the speed and flexibility of web frontend rendering and massive-scale data rendering of the backend.

Arctern 0.1.0 is compatible with Sulidae. The following figures show the visualization effects of a headmap and choropleth with 10 million data.

<img src="./doc/img/v0.1.0_intro/heat_map.png" width = "700">

<img src="./doc/img/v0.1.0_intro/contour_map.png" width = "700">

## Arctern roadmap

### v0.1.0

1. Support most frequently used 35 GIS APIs in the OGC standard.
2. Support rendering horopleths, heatmaps, and scatter plots for massive-scale datasets.
3. Provide C++, Python, and Spark APIs with the Arrow standard.
4. Arctern engine with CPU based implementation.
5. Arctern engine with GPU based implementation.
6. Compatibility with Sulidae.
7. Documentation for installation, deployment, and API reference.

### v0.2.0

1. Domain-specific API for trace analysis and geospatial data statistics.
2. Geospatial indexes for domain-specific API.
3. Performance optimization for Spark 3.0.
4. Support more GIS APIs.
5. Continuously improve system stability.

### In progress:

#### Completed by 2020.03.10

1. Support most frequently used 35 GIS APIs in the OGC standard.
2. Support rendering horopleths, heatmaps, and scatter plots for massive-scale datasets.
3. Support C++, Python, and Spark APIs based the Arrow standard.
4. Arctern engine with CPU support.
5. Arctern engine with GPU support.

### Contact us

#### Email

support@zilliz.com

##### ZILLIZ Wechat

<img src="./doc/img/v0.1.0_intro/zilliz.png" width = "200" align=left>
