### 现有的 gis 函数须依据输出结果对比验证标准不同进行分类，主要有如下几种评价维度：
  - 输出结果为值类型 (表示为 `value`)
  - 输出结果为 wkt 类型 (表示为 `wkt`)
  - 输出结果为普通 string 类型 (表示为 `str`)
  - 精确 (表示为 `precise`)
  - 非精确 (表示为 `unprecise`)
  - 多重误差标准 (表示为 `multi`)
  - 非多重误差标准 (表示为 `single`)

函数分类如下 :
- value+precise (比较方式 : == )
  ```
   ST_Equals
   ST_Touches
   ST_Overlaps
   ST_Crosses
   ST_Contains
   ST_Intersects;
   ST_Within
   ST_IsValid
   ST_IsSimple
   ST_NPoints
  ```
- str+precise (比较方式 : == )
  ```
   ST_GeometryType

  ```
- wkt+precise  (比较方式 : shapely_equal(0) , curve(e-8) , surface(e-8))
  ```
   ST_Point
   ST_PolygonFromEnvelope
   ST_GeomFromGeoJSON
   ST_Union_Aggr
   ST_PrecisionReduce
   ST_MakeValid
   ST_Transform
   ST_SimplifyPreserveTopology
  ```

- wkt+unprecise  (比较方式 : shapely_equal_extract(e-8) , curve(e-2) , surface(e-2))
  ```
  ST_Envelope
  ST_Envelope_Aggr
  ST_Buffer
  ST_Intersection
  ST_Centroid ?
  ST_ConvexHull

  ```    
- value+unprecise+multi (比较方式 : float_diff（按照输入类型指定精度）)
 ```
  ST_Distance
  ST_Area
  ST_Length
  ST_HausdorffDistance
 ```
