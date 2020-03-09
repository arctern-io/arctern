drop table if exists test_table;
create table test_table (geos geometry);
copy test_table from '@path@/convexhull.csv' with delimiter '|' csv header;
SELECT ST_AsText(ST_ConvexHull(geos)) from test_table;
-- drop table if exists test_table;