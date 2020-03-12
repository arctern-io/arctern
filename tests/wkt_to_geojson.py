import geojson
import shapely.wkt
# from shapely.geometry import 

s = '''POLYGON ((23.314208 37.768469, 24.039306 37.768469, 24.039306 38.214372, 23.314208 38.214372, 23.314208 37.768469))'''


def to_geojson(wkt):
    # Convert to a shapely.geometry.polygon.Polygon object
    g1 = shapely.wkt.loads(wkt)
    print(g1)
    g2 = geojson.Feature(geometry=g1, properties={})
    return g2.geometry

with open('./geojson.txt', 'r') as f:
    lines = f.readlines()[1:]
    print(lines[0])
    jsons = [to_geojson(l) for l in lines]

    for j in jsons:
        print(j)

# w = "MULTIPOLYGON (((1 1,1 2,2 2,2 1,1 1)),((0 0,1 -1,1 1,-2 3,0 0)))"
# print(to_geojson(w))