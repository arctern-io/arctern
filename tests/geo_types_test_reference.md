postgis sql :
```sql
drop table if exists test_table;
create table test_table (geos geometry);
insert into test_table values 
('POINT (1 8)'),
('MULTIPOINT (1 1,3 4)'),
('LINESTRING (1 1,1 2,2 3)'),
('MULTILINESTRING ((1 1,1 2),(2 4,1 9,1 8))'), 
('MULTILINESTRING ((1 1,3 4))'),
('POLYGON ((1 1,1 2,2 2,2 1,1 1))'),
-- ('POLYGON ((1 1,1 2,2 2,2 1,1 1)),((0 0,1 -1,3 4,-2 3,0 0))'), 
('POLYGON ((1 1,1 2,2 2,2 1,1 1),(0 0,1 -1,3 4,-2 3,0 0))'),   
('MULTIPOLYGON (((1 1,1 2,2 2,2 1,1 1)),((0 0,1 -1,3 4,-2 3,0 0)) )'),

('POINT EMPTY'),
('LINESTRING EMPTY'),
('POLYGON EMPTY'),
('MULTIPOINT EMPTY'),
('MULTILINESTRING EMPTY'),
('MULTIPOLYGON EMPTY'),
('GEOMETRYCOLLECTION EMPTY'),

('CIRCULARSTRING (0 2, -1 1,0 0, 0.5 0, 1 0, 2 1, 1 2, 0.5 2, 0 2)'),

('COMPOUNDCURVE(CIRCULARSTRING(0 2, -1 1,0 0),(0 0, 0.5 0, 1 0),CIRCULARSTRING( 1 0, 2 1, 1 2),(1 2, 0.5 2, 0 2))'),

('GEOMETRYCOLLECTION ( LINESTRING ( 90 190, 120 190, 50 60, 130 10, 190 50, 160 90, 10 150, 90 190 ), POINT(90 190) ) '),

('MULTICURVE ((5 5, 3 5, 3 3, 0 3), CIRCULARSTRING (0 0, 0.2 1, 0.5 1.4), COMPOUNDCURVE (CIRCULARSTRING (0 0,1 1,1 0),(1 0,0 1)))'),

('CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0),(1 1, 3 3, 3 1, 1 1))'),

('CURVEPOLYGON(COMPOUNDCURVE(CIRCULARSTRING(0 0,2 0, 2 1, 2 3, 4 3),(4 3, 4 5, 1 4, 0 0)), CIRCULARSTRING(1.7 1, 1.4 0.4, 1.6 0.4, 1.6 0.5, 1.7 1) )'),

('MULTISURFACE(CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0),(1 1, 3 3, 3 1, 1 1)),((10 10, 14 12, 11 10, 10 10),(11 11, 11.5 11, 11 11.5, 11 11)))'),

('MULTISURFACE Z (CURVEPOLYGON Z (CIRCULARSTRING Z (-2 0 0, -1 -1 1, 0 0 2, 1 -1 3, 2 0 4, 0 2 2, -2 0 0), (-1 0 1, 0 0.5 2, 1 0 3, 0 1 3, -1 0 1)), ((7 8 7, 10 10 5, 6 14 3, 4 11 4, 7 8 7)))'),

('MULTISURFACE (CURVEPOLYGON (CIRCULARSTRING (-2 0, -1 -1, 0 0, 1 -1, 2 0, 0 2, -2 0), (-1 0, 0 0.5, 1 0, 0 1, -1 0)), ((7 8, 10 10, 6 14, 4 11, 7 8)))'),

('POLYHEDRALSURFACE (((0 0,0 0,0 1,0 0)),((0 0,0 1,1 0,0 0)),((0 0,1 0,0 0,0 0)),((1 0,0 1,0 0,1 0)))'),

('TRIANGLE ((1 2,4 5,7 8,1 2))'),

('TIN ( ((0 0, 0 0, 0 1, 0 0)), ((0 0, 0 1, 1 1, 0 0)) )')
;

select st_area(geos) from test_table;
```

arctern test code : 
```python
def run_st_tmp(spark):
    register_funcs(spark)
    input = []

    input.extend([('POINT (1 8)',)])
    input.extend([('MULTIPOINT (1 1,3 4)',)])
    input.extend([('LINESTRING (1 1,1 2,2 3)',)])
    input.extend([('MULTILINESTRING ((1 1,1 2),(2 4,1 9,1 8))',)])
    input.extend([('MULTILINESTRING ((1 1,3 4))',)])
    input.extend([('POLYGON ((1 1,1 2,2 2,2 1,1 1))',)])
    #input.extend([('POLYGON ((1 1,1 2,2 2,2 1,1 1)),((0 0,1 -1,3 4,-2 3,0 0))',)])
    input.extend([('POLYGON ((1 1,1 2,2 2,2 1,1 1),(0 0,1 -1,3 4,-2 3,0 0))',)])
    input.extend([('MULTIPOLYGON (((1 1,1 2,2 2,2 1,1 1)),((0 0,1 -1,3 4,-2 3,0 0)) )',)])
    input.extend([('POINT EMPTY',)])
    input.extend([('LINESTRING EMPTY',)])
    input.extend([('POLYGON EMPTY',)])
    input.extend([('MULTIPOINT EMPTY',)])
    input.extend([('MULTILINESTRING EMPTY',)])
    input.extend([('MULTIPOLYGON EMPTY',)])
    input.extend([('GEOMETRYCOLLECTION EMPTY',)])
    input.extend([('CIRCULARSTRING (0 2, -1 1,0 0, 0.5 0, 1 0, 2 1, 1 2, 0.5 2, 0 2)',)])
    input.extend([('COMPOUNDCURVE(CIRCULARSTRING(0 2, -1 1,0 0),(0 0, 0.5 0, 1 0),CIRCULARSTRING( 1 0, 2 1, 1 2),(1 2, 0.5 2, 0 2))',)])
    input.extend([('GEOMETRYCOLLECTION ( LINESTRING ( 90 190, 120 190, 50 60, 130 10, 190 50, 160 90, 10 150, 90 190 ), POINT(90 190) ) ',)])
    input.extend([('MULTICURVE ((5 5, 3 5, 3 3, 0 3), CIRCULARSTRING (0 0, 0.2 1, 0.5 1.4), COMPOUNDCURVE (CIRCULARSTRING (0 0,1 1,1 0),(1 0,0 1)))',)])
    input.extend([('CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0),(1 1, 3 3, 3 1, 1 1))',)])
    input.extend([('CURVEPOLYGON(COMPOUNDCURVE(CIRCULARSTRING(0 0,2 0, 2 1, 2 3, 4 3),(4 3, 4 5, 1 4, 0 0)), CIRCULARSTRING(1.7 1, 1.4 0.4, 1.6 0.4, 1.6 0.5, 1.7 1) )',)])
    input.extend([('MULTISURFACE(CURVEPOLYGON(CIRCULARSTRING(0 0, 4 0, 4 4, 0 4, 0 0),(1 1, 3 3, 3 1, 1 1)),((10 10, 14 12, 11 10, 10 10),(11 11, 11.5 11, 11 11.5, 11 11)))',)])
    input.extend([('MULTISURFACE Z (CURVEPOLYGON Z (CIRCULARSTRING Z (-2 0 0, -1 -1 1, 0 0 2, 1 -1 3, 2 0 4, 0 2 2, -2 0 0), (-1 0 1, 0 0.5 2, 1 0 3, 0 1 3, -1 0 1)), ((7 8 7, 10 10 5, 6 14 3, 4 11 4, 7 8 7)))',)])
    input.extend([('MULTISURFACE (CURVEPOLYGON (CIRCULARSTRING (-2 0, -1 -1, 0 0, 1 -1, 2 0, 0 2, -2 0), (-1 0, 0 0.5, 1 0, 0 1, -1 0)), ((7 8, 10 10, 6 14, 4 11, 7 8)))',)])
    input.extend([('POLYHEDRALSURFACE (((0 0,0 0,0 1,0 0)),((0 0,0 1,1 0,0 0)),((0 0,1 0,0 0,0 0)),((1 0,0 1,0 0,1 0)))',)])
    input.extend([('TRIANGLE ((1 2,4 5,7 8,1 2))',)])
    input.extend([('TIN ( ((0 0, 0 0, 0 1, 0 0)), ((0 0, 0 1, 1 1, 0 0)) )',)])

    df = spark.createDataFrame(data=input, schema=['geos']).cache()
    df.createOrReplaceTempView("t1")
    spark.sql("select ST_Area_UDF(geos) from t1").show(100,0)
```

- postgis result :
```
                0
                0
                0
                0
                0
                1
              -11
               13
                0
                0
                0
                0
                0
                0
                0
                0
                0
                0
                0
  23.122649255638
 9.05303418413299
  23.997649255638
 26.4209934708643
 26.4209934708643
                1
                0
              0.5
```

- arctern result :
```
 0.0                
 0.0                
 0.0                
 0.0                
 0.0                
 1.0                
 -11.0              
 13.0               
 0.0                
 0.0                
 0.0                
 0.0                
 0.0                
 0.0                
 0.0                
 5.139041318485766  
 5.139041318485766  
 4950.0             
 0.0                
 23.132741228718352 
 9.05354803115023   
 24.007741228718352 
 26.417123955457246 
 26.417123955457246 
 -1.0               
 0.0                
 -1.0        
```

