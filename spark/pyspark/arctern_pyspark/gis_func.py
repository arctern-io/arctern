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
    "union_aggr",
    "envelope_aggr",
]

def _agg_func_template(df, col_name, st_agg_func):
    import pandas as pd
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    from pyspark.sql.types import (StructType, StructField, BinaryType)

    agg_schema = StructType([StructField('geos', BinaryType(), True)])
    @pandas_udf(agg_schema, PandasUDFType.MAP_ITER)
    def agg_step1(batch_iter, col_name=col_name):
        for pdf in batch_iter:
            ret = st_agg_func(pdf[col_name])
            df = pd.DataFrame({"geos": [ret[0]]})
            yield df

    @pandas_udf(BinaryType(), PandasUDFType.GROUPED_AGG)
    def agg_step2(geos):
        return st_agg_func(geos)[0]

    agg_df = df.mapInPandas(agg_step1)
    ret = agg_df.agg(agg_step2(agg_df['geos'])).collect()[0][0]
    return ret

def union_aggr(df, col_name):
    from arctern import ST_Union_Aggr
    return _agg_func_template(df, col_name, ST_Union_Aggr)

def envelope_aggr(df, col_name):
    from arctern import ST_Envelope_Aggr
    return _agg_func_template(df, col_name, ST_Envelope_Aggr)
