# GIS

#### ST_Intersection

** Introduction: Return the intersection shape of two geometries. The return type is a geometry. The data in geometries is organized into WTK.**

*Format: std::shared_ptr\<arrow::Array> ST_Intersection (std::shared_ptr\<arrow::Array> geometries_1, std::shared_ptr\<arrow::Array> geometries_2)*


#### ST_IsValid

** Introduction: Test if Geometry is valid.Return an array of boolean,geometries_1 is an array of WKT. **

*Format: std::shared_ptr\<arrow::Array> ST_IsValid (std::shared_ptr\<arrow::Array> geometries_1)*


#### ST_PrecisionReduce

** Introduction: Reduce the precision of the given geometry to the given number of decimal places.Return an array of WTK,geometies_1 is an array of WKT. **

*Format: std::shared_ptr\<arrow::Array> ST_PrecisionReduce (std::shared_ptr\<arrow::Array> geometies_1, int32_t num_dot)*

#### ST_Equals

** Introduction: Test if leftGeometry is equal to rightGeometry.Return an array of boolean,The data in geometries is organized into WTK. **

*Format: std::shared_ptr\<arrow::Array> ST_Equals (std::shared_ptr\<arrow::Array> geometries_1, std::shared_ptr\<arrow::Array> geometries_2)*


#### ST_Touches

** Introduction: Test if leftGeometry touches rightGeometry.Return an array of boolean,The data in geometries is organized into WTK.**

*Format: std::shared_ptr\<arrow::Array> ST_Touches (std::shared_ptr\<arrow::Array> geometries_1, std::shared_ptr\<arrow::Array> geometries_2)*


#### ST_Overlaps

** Introduction: Test if leftGeometry overlaps rightGeometry.Return an array of boolean,The data in geometries is organized into WTK. **

*Format: std::shared_ptr\<arrow::Array> ST_Touches (std::shared_ptr\<arrow::Array> geometries_1, std::shared_ptr\<arrow::Array> geometries_2)*


#### ST_Crosses

** Introduction: Test if leftGeometry crosses rightGeometry.Return an array of boolean,The data in geometries is organized into WTK. **

*Format: std::shared_ptr\<arrow::Array> ST_Crosses (std::shared_ptr\<arrow::Array> geometries_1, std::shared_ptr\<arrow::Array> geometries_2)*


#### ST_IsSimple

** Introduction: Test if Geometry is simple.Return an array of boolean,The data in geometries is organized into WTK. **

*Format: std::shared_ptr\<arrow::Array> ST_IsSimple (std::shared_ptr\<arrow::Array> geometries)*


ST_MakeValid
** Introduction: Given an invalid polygon or multipolygon and removeHoles boolean flag, create a valid representation of the geometry. **

Format: ST_MakeValid (A:geometry, removeHoles:Boolean)

Spark SQL Example:

SELECT geometryValid.polygon FROM table
LATERAL VIEW ST_MakeValid(polygon, false) geometryValid AS polygon


ST_SimplifyPreserveTopology
ST_AsText
ST_GeometryType 
