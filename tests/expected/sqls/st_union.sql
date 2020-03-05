drop table if exists test_table;
create table test_table (geos geometry);
insert into test_table values
('MULTILINESTRING ((-5 45, 8 36), (1 49, 15 41))'),
('POLYGON ((0 0,4 0,4 4,0 4,0 0))'),
('POLYGON ((5 1,7 1,7 2,5 2,5 1))'),
('MULTIPOINT ((0 0), (10 0))');

select st_astext(st_union(geos)) from test_table;
