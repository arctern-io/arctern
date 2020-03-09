drop table if exists test_envelope;
create table test_envelope (geos geometry);
insert into test_envelope values 
('POINT (1 8)'),
('MULTIPOINT (1 1,3 4)'),
('LINESTRING (1 1,1 2,2 3)'),
('MULTILINESTRING ((1 1,1 2),(2 4,1 9,1 8))'), 
('MULTILINESTRING ((1 1,3 4))'),
('POLYGON ((1 1,1 2,2 2,2 1,1 1))'),
('POINT EMPTY'),
('LINESTRING EMPTY'),
('POLYGON EMPTY'),
('MULTIPOINT EMPTY'),
('MULTILINESTRING EMPTY'),
('MULTIPOLYGON EMPTY'),
('GEOMETRYCOLLECTION EMPTY'),
('CIRCULARSTRING (0 2, -1 1,0 0, 0.5 0, 1 0, 2 1, 1 2, 0.5 2, 0 2)'),
('COMPOUNDCURVE(CIRCULARSTRING(0 2, -1 1,0 0),(0 0, 0.5 0, 1 0),CIRCULARSTRING( 1 0, 2 1, 1 2),(1 2, 0.5 2, 0 2))')
;

select st_astext(st_union(st_envelope(geos))) from test_envelope;
