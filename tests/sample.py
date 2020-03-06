import shapely
from shapely import wkt

filename = './expected/results/st_envelope.out'
# filename = './data/envelope.json'
# filename = './arctern_results/run_test_st_envelope.json'

#g1 = 'GEOMETRYCOLLECTION (POINT (10 0), LINESTRING (-5 45, 8 36), LINESTRING (1 49, 15 41), POLYGON ((0 0, 0 4, 4 4, 4 0, 0 0)), POLYGON ((5 1, 5 2, 7 2, 7 1, 5 1)))'
#g2 = 'GEOMETRYCOLLECTION (POINT (10 0), LINESTRING (-5 45, 8 36), LINESTRING (1 49, 15 41), POLYGON ((5 1, 7 1, 7 2, 5 2, 5 1)), POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0)))'

#g1 = "MULTILINESTRING((0 0, 1 1),(0 0, 1 1, 2 2))"
#g2 = "MULTILINESTRING((0 0, 1 1, 2 2),(0 0, 1 1))"

g1 = "MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0),(5 5, 5 6, 6 6, 8 5, 5 5)),((100 100, 100 130, 130 130, 130 100, 100 100)))"
g2 = "MULTIPOLYGON (((100 100, 100 130, 130 130, 130 100, 100 100)), ((0 0, 10 0, 10 10, 0 10, 0 0),(5 5, 5 6, 6 6, 8 5, 5 5)))"

geo1 = list(wkt.loads(g1))
geo2 = list(wkt.loads(g2))
print(type(geo1))

geo1.sort()
geo2.sort()
print(geo1 == geo2)
exit(0)
geo1 = wkt.loads(g1)
geo2 = wkt.loads(g2)

print(geo1.type)
print(list(geo1))
print(geo1.equals(geo2))
print(geo1.equals_exact(geo2, 0.001))

print(dir(geo1))
exit(0)






with open(filename, 'r') as f:
    lines = f.readlines()
    print(lines)
    print(len(lines))
    for line in lines:
        # print(wkt.loads(line))
        print(line)
