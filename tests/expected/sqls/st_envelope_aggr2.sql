drop table if exists test_envelope;
create table test_envelope (x float, y float);
insert into test_envelope values 
(8, 10),
(8, 6),
(10, 1),
(5, 10),
(3, 5),
(7, 3),
(45.747872, 96.193504),
(21.507721, 37.289151),
(73.388003, 81.457667),
(52.096912, 21.554577),
(80.335055, 10.929492),
(51.879734, 16.702802);

select st_astext(st_envelope(geos)) from (select st_union(st_point(x,y)) as geos from test_envelope) as envelope_aggr;
