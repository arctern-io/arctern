import geojson
import shapely.wkt
# from shapely.geometry import

# pylint: disable=redefined-outer-name

s = '''POLYGON ((23.314208 37.768469, 24.039306 37.768469, 24.039306 38.214372, 23.314208 38.214372, 23.314208 37.768469))'''

ss = '''POLYGON ((0 0,0.0 111325.142866385,111319.490793274 111325.142866385,111319.490793274 0.0,0 0),(0.0 111325.142866385,111319.490793274 111325.142866385,111319.490793274 222684.208505545,0.0 222684.208505545,0.0 111325.142866385))'''

ss = 'POLYGON ((0 0,0 1,1 1,1 0,0 0),(0 1,1 1,1 2,0 2,0 1))'

def to_geojson(wkt):
    # Convert to a shapely.geometry.polygon.Polygon object
    g1 = shapely.wkt.loads(wkt)
    print(g1)
    g2 = geojson.Feature(geometry=g1, properties={})
    return g2.geometry

def all_geojson():
    with open('./geojson.txt', 'r') as f:
        lines = f.readlines()[1:]
        print(lines[0])
        jsons = [to_geojson(l) for l in lines]

    for j in jsons:
        print(j)

def load_wkt(wkt):
    g = shapely.wkt.loads(wkt)
    return g
    # print(type(g))
    # print(g)

if __name__ == '__main__':
    with open('/tmp/arctern_results/run_test_st_transform.csv', 'r') as f:
        lines = f.readlines()
        # arr = [load_wkt(x) for x in lines]
    # for x in lines:
    #     print(x.strip())

    with open('bf.csv', 'r') as f:
        lines = f.readlines()
        res = [x for x in lines if not x.startswith('xxxxx')]

    with open('bfnew.csv', 'w') as f:
        for x in res:
            f.writelines(x)
    import sys
    sys.exit()
    #with open('./expected/results/st_transform.out', 'r') as f:
    #    lines = f.readlines()
    #    brr = [load_wkt(x) for x in lines]

    #for a, b in zip(arr, brr):
    #    print(a.equals(b))
# w = "MULTIPOLYGON (((1 1,1 2,2 2,2 1,1 1)),((0 0,1 -1,1 1,-2 3,0 0)))"
# print(to_geojson(w))
