import shapely
from shapely import wkt
from osgeo import ogr

g1 = 'POINT (3339584.72379821 1118889.97485796)'
g2 = 'POINT (3339584.72379811 1118889.97485792)'

geo1 = wkt.loads(g1)
geo2 = wkt.loads(g2)

print(geo1.distance(geo2))