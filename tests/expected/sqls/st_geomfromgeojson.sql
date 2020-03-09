drop table if exists test_table;
create table test_table (geos text);
copy test_table from '....../geojson.csv' with delimiter '|' csv header;
select st_astext(ST_GeomFromGeoJSON(geos)) from test_table;
drop table if exists test_table;
