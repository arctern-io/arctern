# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "pointmap",
    "weighted_pointmap",
    "heatmap",
    "choroplethmap",
    "icon_viz",
    "fishnetmap",
]

def pointmap(vega, df):
    if df.rdd.isEmpty():
        return None

    if len(df.schema.names) != 1:
        return None

    col_point = df.schema.names[0]

    from pyspark.sql.functions import pandas_udf, PandasUDFType, col, lit
    from ._wrapper_func import TransformAndProjection, Projection

    bounding_box = vega.bounding_box()
    top_left = 'POINT (' + str(bounding_box[0]) + ' ' + str(bounding_box[3]) + ')'
    bottom_right = 'POINT (' + str(bounding_box[2]) + ' ' + str(bounding_box[1]) + ')'

    height = vega.height()
    width = vega.width()
    coor = vega.coor()

    if coor != 'EPSG:3857':
        df = df.select(TransformAndProjection(col(col_point), lit(str(coor)), lit('EPSG:3857'), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_point))
    else:
        df = df.select(Projection(col(col_point), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_point))

    @pandas_udf("string", PandasUDFType.GROUPED_AGG)
    def pointmap_wkb(point, conf=vega):
        from arctern import point_map_layer
        return point_map_layer(conf, point, False)

    df = df.rdd.coalesce(1, shuffle=True).toDF()
    hex_data = df.agg(pointmap_wkb(df[col_point])).collect()[0][0]
    return hex_data


# pylint: disable=too-many-statements
def weighted_pointmap(vega, df):
    if df.rdd.isEmpty():
        return None

    if len(df.schema.names) == 1:
        col_point = df.schema.names[0]
        render_mode = 0
    elif len(df.schema.names) == 2:
        col_point = df.schema.names[0]
        col_count = df.schema.names[1]
        render_mode = 1
    elif len(df.schema.names) == 3:
        col_point = df.schema.names[0]
        col_color = df.schema.names[1]
        col_stroke = df.schema.names[2]
        render_mode = 2
    else:
        return None

    from pyspark.sql.functions import pandas_udf, PandasUDFType, col, lit
    from pyspark.sql.types import (StructType, StructField, BinaryType, IntegerType)
    from ._wrapper_func import TransformAndProjection, Projection

    bounding_box = vega.bounding_box()
    top_left = 'POINT (' + str(bounding_box[0]) + ' ' + str(bounding_box[3]) + ')'
    bottom_right = 'POINT (' + str(bounding_box[2]) + ' ' + str(bounding_box[1]) + ')'

    height = vega.height()
    width = vega.width()
    coor = vega.coor()
    aggregation_type = vega.aggregation_type()

    if coor == 'EPSG:3857':
        if render_mode == 2:
            df = df.select(Projection(col(col_point), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_point), col(col_color), col(col_stroke))
            agg_schema = StructType([StructField(col_point, BinaryType(), True),
                                     StructField(col_color, IntegerType(), True),
                                     StructField(col_stroke, IntegerType(), True)])

            from typing import Iterator
            import pandas
            def render_agg_UDF_3857_2(batch_iter: Iterator[pandas.DataFrame]) -> Iterator[pandas.DataFrame]:
                for pdf in batch_iter:
                    dd = pdf.groupby([col_point])
                    ll = [col_color, col_stroke]
                    dd = dd[ll].agg([aggregation_type]).reset_index()
                    dd.columns = [col_point, col_color, col_stroke]
                    yield dd

            @pandas_udf("string", PandasUDFType.GROUPED_AGG)
            def weighted_pointmap_wkb_3857_2(point, c, s, conf=vega):
                from arctern import weighted_point_map_layer
                return weighted_point_map_layer(conf, point, False, color_weights=c, size_weights=s)

            agg_df = df.mapInPandas(render_agg_UDF_3857_2, schema=agg_schema)
            agg_df = agg_df.rdd.coalesce(1, shuffle=True).toDF()
            hex_data = agg_df.agg(weighted_pointmap_wkb_3857_2(agg_df[col_point], agg_df[col_color], agg_df[col_stroke])).collect()[0][0]
        elif render_mode == 1:
            df = df.select(Projection(col(col_point), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_point), col(col_count))
            agg_schema = StructType([StructField(col_point, BinaryType(), True),
                                     StructField(col_count, IntegerType(), True)])

            from typing import Iterator
            import pandas
            def render_agg_UDF_3857_1(batch_iter: Iterator[pandas.DataFrame]) -> Iterator[pandas.DataFrame]:
                for pdf in batch_iter:
                    dd = pdf.groupby([col_point])
                    dd = dd[col_count].agg([aggregation_type]).reset_index()
                    dd.columns = [col_point, col_count]
                    yield dd

            @pandas_udf("string", PandasUDFType.GROUPED_AGG)
            def weighted_pointmap_wkb_3857_1(point, c, conf=vega):
                from arctern import weighted_point_map_layer
                return weighted_point_map_layer(conf, point, False, color_weights=c)

            agg_df = df.mapInPandas(render_agg_UDF_3857_1, schema=agg_schema)
            agg_df = agg_df.rdd.coalesce(1, shuffle=True).toDF()
            hex_data = agg_df.agg(weighted_pointmap_wkb_3857_1(agg_df[col_point], agg_df[col_count])).collect()[0][0]
        else:
            df = df.select(Projection(col(col_point), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_point))
            @pandas_udf("string", PandasUDFType.GROUPED_AGG)
            def weighted_pointmap_wkb(point, conf=vega):
                from arctern import weighted_point_map_layer
                return weighted_point_map_layer(conf, point, False)

            df = df.rdd.coalesce(1, shuffle=True).toDF()
            hex_data = df.agg(weighted_pointmap_wkb(df[col_point])).collect()[0][0]
        return hex_data

    if render_mode == 2:
        df = df.select(TransformAndProjection(col(col_point), lit(str(coor)), lit('EPSG:3857'), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_point), col(col_color), col(col_stroke))
        agg_schema = StructType([StructField(col_point, BinaryType(), True),
                                 StructField(col_color, IntegerType(), True),
                                 StructField(col_stroke, IntegerType(), True)])

        from typing import Iterator
        import pandas
        def render_agg_UDF_2(batch_iter: Iterator[pandas.DataFrame]) -> Iterator[pandas.DataFrame]:
            for pdf in batch_iter:
                dd = pdf.groupby([col_point])
                ll = [col_color, col_stroke]
                dd = dd[ll].agg([aggregation_type]).reset_index()
                dd.columns = [col_point, col_color, col_stroke]
                yield dd

        @pandas_udf("string", PandasUDFType.GROUPED_AGG)
        def weighted_pointmap_wkb_2(point, c, s, conf=vega):
            from arctern import weighted_point_map_layer
            return weighted_point_map_layer(conf, point, False, color_weights=c, size_weights=s)

        agg_df = df.mapInPandas(render_agg_UDF_2, schema=agg_schema)
        agg_df = agg_df.rdd.coalesce(1, shuffle=True).toDF()
        hex_data = agg_df.agg(weighted_pointmap_wkb_2(agg_df[col_point], agg_df[col_color], agg_df[col_stroke])).collect()[0][0]
    elif render_mode == 1:
        df = df.select(TransformAndProjection(col(col_point), lit(str(coor)), lit('EPSG:3857'), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_point), col(col_count))
        agg_schema = StructType([StructField(col_point, BinaryType(), True),
                                 StructField(col_count, IntegerType(), True)])

        from typing import Iterator
        import pandas
        def render_agg_UDF_1(batch_iter: Iterator[pandas.DataFrame]) -> Iterator[pandas.DataFrame]:
            for pdf in batch_iter:
                dd = pdf.groupby([col_point])
                dd = dd[col_count].agg([aggregation_type]).reset_index()
                dd.columns = [col_point, col_count]
                yield dd

        @pandas_udf("string", PandasUDFType.GROUPED_AGG)
        def weighted_pointmap_wkb_1(point, c, conf=vega):
            from arctern import weighted_point_map_layer
            return weighted_point_map_layer(conf, point, False, color_weights=c)

        agg_df = df.mapInPandas(render_agg_UDF_1, schema=agg_schema)
        agg_df = agg_df.rdd.coalesce(1, shuffle=True).toDF()
        hex_data = agg_df.agg(weighted_pointmap_wkb_1(agg_df[col_point], agg_df[col_count])).collect()[0][0]
    else:
        df = df.select(TransformAndProjection(col(col_point), lit(str(coor)), lit('EPSG:3857'), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_point))
        @pandas_udf("string", PandasUDFType.GROUPED_AGG)
        def weighted_pointmap_wkb_0(point, conf=vega):
            from arctern import weighted_point_map_layer
            return weighted_point_map_layer(conf, point, False)

        df = df.rdd.coalesce(1, shuffle=True).toDF()
        hex_data = df.agg(weighted_pointmap_wkb_0(df[col_point])).collect()[0][0]
    return hex_data


def heatmap(vega, df):
    if df.rdd.isEmpty():
        return None

    if len(df.schema.names) != 2:
        return None

    col_point = df.schema.names[0]
    col_count = df.schema.names[1]

    from pyspark.sql.functions import pandas_udf, PandasUDFType, lit, col
    from pyspark.sql.types import (StructType, StructField, BinaryType, IntegerType)
    from ._wrapper_func import TransformAndProjection, Projection

    bounding_box = vega.bounding_box()
    top_left = 'POINT (' + str(bounding_box[0]) + ' ' + str(bounding_box[3]) + ')'
    bottom_right = 'POINT (' + str(bounding_box[2]) + ' ' + str(bounding_box[1]) + ')'

    height = vega.height()
    width = vega.width()
    coor = vega.coor()
    aggregation_type = vega.aggregation_type()

    if coor != 'EPSG:3857':
        df = df.select(TransformAndProjection(col(col_point), lit(str(coor)), lit('EPSG:3857'), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_point), col(col_count))
    else:
        df = df.select(Projection(col(col_point), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_point), col(col_count))

    agg_schema = StructType([StructField(col_point, BinaryType(), True),
                             StructField(col_count, IntegerType(), True)])

    import pandas
    from typing import Iterator
    def render_agg_UDF(batch_iter: Iterator[pandas.DataFrame]) -> Iterator[pandas.DataFrame]:
        for pdf in batch_iter:
            dd = pdf.groupby([col_point])
            dd = dd[col_count].agg([aggregation_type]).reset_index()
            dd.columns = [col_point, col_count]
            yield dd

    @pandas_udf("string", PandasUDFType.GROUPED_AGG)
    def heatmap_wkb(point, w, conf=vega):
        from arctern import heat_map_layer
        return heat_map_layer(conf, point, w, False)

    agg_df = df.mapInPandas(render_agg_UDF, schema=agg_schema)
    agg_df = agg_df.rdd.coalesce(1, shuffle=True).toDF()
    hex_data = agg_df.agg(heatmap_wkb(agg_df[col_point], agg_df[col_count])).collect()[0][0]
    return hex_data


def choroplethmap(vega, df):
    if df.rdd.isEmpty():
        return None

    if len(df.schema.names) != 2:
        return None

    col_polygon = df.schema.names[0]
    col_count = df.schema.names[1]

    from pyspark.sql.functions import pandas_udf, PandasUDFType, col, lit
    from pyspark.sql.types import (StructType, StructField, BinaryType, IntegerType)
    from ._wrapper_func import TransformAndProjection, Projection

    bounding_box = vega.bounding_box()
    top_left = 'POINT (' + str(bounding_box[0]) + ' ' + str(bounding_box[3]) + ')'
    bottom_right = 'POINT (' + str(bounding_box[2]) + ' ' + str(bounding_box[1]) + ')'

    height = vega.height()
    width = vega.width()
    coor = vega.coor()
    aggregation_type = vega.aggregation_type()

    if coor != 'EPSG:3857':
        df = df.select(TransformAndProjection(col(col_polygon), lit(str(coor)), lit('EPSG:3857'), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_polygon), col(col_count))
    else:
        df = df.select(Projection(col(col_polygon), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_polygon), col(col_count))

    agg_schema = StructType([StructField(col_polygon, BinaryType(), True),
                             StructField(col_count, IntegerType(), True)])

    from typing import Iterator
    import pandas
    def render_agg_UDF(batch_iter: Iterator[pandas.DataFrame]) -> Iterator[pandas.DataFrame]:
        for pdf in batch_iter:
            dd = pdf.groupby([col_polygon])
            dd = dd[col_count].agg([aggregation_type]).reset_index()
            dd.columns = [col_polygon, col_count]
            yield dd

    @pandas_udf("string", PandasUDFType.GROUPED_AGG)
    def choroplethmap_wkb(wkb, w, conf=vega):
        from arctern import choropleth_map_layer
        return choropleth_map_layer(conf, wkb, w, False)

    agg_df = df.mapInPandas(render_agg_UDF, schema=agg_schema)
    agg_df = agg_df.rdd.coalesce(1, shuffle=True).toDF()
    hex_data = agg_df.agg(choroplethmap_wkb(agg_df[col_polygon], agg_df[col_count])).collect()[0][0]
    return hex_data


def icon_viz(vega, df):
    if df.rdd.isEmpty():
        return None

    if len(df.schema.names) != 1:
        return None

    col_point = df.schema.names[0]

    from pyspark.sql.functions import pandas_udf, PandasUDFType, col, lit
    from ._wrapper_func import TransformAndProjection, Projection

    bounding_box = vega.bounding_box()
    top_left = 'POINT (' + str(bounding_box[0]) + ' ' + str(bounding_box[3]) + ')'
    bottom_right = 'POINT (' + str(bounding_box[2]) + ' ' + str(bounding_box[1]) + ')'

    height = vega.height()
    width = vega.width()
    coor = vega.coor()

    if coor != 'EPSG:3857':
        df = df.select(TransformAndProjection(col(col_point), lit(str(coor)), lit('EPSG:3857'), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_point))
    else:
        df = df.select(Projection(col(col_point), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_point))

    @pandas_udf("string", PandasUDFType.GROUPED_AGG)
    def iconviz(point, conf=vega):
        from arctern import icon_viz_layer
        return icon_viz_layer(conf, point, False)

    df = df.rdd.coalesce(1, shuffle=True).toDF()
    hex_data = df.agg(iconviz(df[col_point])).collect()[0][0]
    return hex_data

def fishnetmap(vega, df):
    if df.rdd.isEmpty():
        return None

    if len(df.schema.names) != 2:
        return None

    col_point = df.schema.names[0]
    col_count = df.schema.names[1]

    from pyspark.sql.functions import pandas_udf, PandasUDFType, lit, col
    from pyspark.sql.types import (StructType, StructField, BinaryType, IntegerType)
    from ._wrapper_func import TransformAndProjection, Projection

    bounding_box = vega.bounding_box()
    top_left = 'POINT (' + str(bounding_box[0]) + ' ' + str(bounding_box[3]) + ')'
    bottom_right = 'POINT (' + str(bounding_box[2]) + ' ' + str(bounding_box[1]) + ')'

    height = vega.height()
    width = vega.width()
    coor = vega.coor()
    aggregation_type = vega.aggregation_type()

    if coor != 'EPSG:3857':
        df = df.select(TransformAndProjection(col(col_point), lit(str(coor)), lit('EPSG:3857'), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_point), col(col_count))
    else:
        df = df.select(Projection(col(col_point), lit(bottom_right), lit(top_left), lit(int(height)), lit(int(width))).alias(col_point), col(col_count))

    agg_schema = StructType([StructField(col_point, BinaryType(), True),
                             StructField(col_count, IntegerType(), True)])

    from typing import Iterator
    import pandas
    def render_agg_UDF(batch_iter: Iterator[pandas.DataFrame]) -> Iterator[pandas.DataFrame]:
        for pdf in batch_iter:
            dd = pdf.groupby([col_point])
            dd = dd[col_count].agg([aggregation_type]).reset_index()
            dd.columns = [col_point, col_count]
            yield dd

    @pandas_udf("string", PandasUDFType.GROUPED_AGG)
    def fishnetmap_wkb(point, w, conf=vega):
        from arctern import fishnet_map_layer
        return fishnet_map_layer(conf, point, w, False)

    agg_df = df.mapInPandas(render_agg_UDF, schema=agg_schema)
    agg_df = agg_df.rdd.coalesce(1, shuffle=True).toDF()
    hex_data = agg_df.agg(fishnetmap_wkb(agg_df[col_point], agg_df[col_count])).collect()[0][0]
    return hex_data
