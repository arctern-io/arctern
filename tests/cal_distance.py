import shapely
from shapely import wkt
from osgeo import ogr
from ogr import *

EPOCH = 1e-6

geo_types = ['POLYGON', 'POINT', 'LINESTRING']
geo_collection_types = ['MULTIPOLYGON', 'MULTIPOINT', 'MULTILINESTRING', 'GEOMETRYCOLLECTION']

def is_empty(geo):
    geo = geo.strip().upper()
    if geo.endswith('EMPTY'):
        return True
    else:
        return False

def is_point(geo):
    geo = geo.strip().upper()

    if geo.startswith('POINT'):
        return True
    else:
        return False
    
def is_linestring(geo):
    geo = geo.strip().upper()

    if geo.startswith('LINESTRING'):
        return True
    else:
        return False

def is_polygon(geo):
    geo = geo.strip().upper()

    if geo.startswith('POLYGON'):
        return True
    else:
        return False

def is_geometry(geo):
    geo = geo.strip().upper()

    for x in geo_types:
        if geo.startswith(x) and len(geo) != len(x):
            return True
        else:
            continue
    
    return False

def is_geometrycollection(geo):
    geo = geo.strip().upper()

    for x in geo_collection_types:
        if geo.startswith(x):
            return True
        else:
            continue
    
    return False

g1 = 'POINT (0 0)'
g2 = 'LINESTRING(0.0754083271833685 -0.0717946957988853,0.154248620486045 -0.139802800502636,0.236330946735871 -0.203860476630213,0.321457562419346 -0.263813403629037,0.409423390063079 -0.319517149646711,0.500016512283379 -0.370837519480006,0.593018682313797 -0.41765087786286,0.688205849780703 -0.459844447314564)'
g3 = 'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'
g4 = 'POLYGON ((1 0, 1 1, 2 1, 2 0, 1 0))'
g5 = 'MULTIPOLYGON (((0 0,445277.963173094 0.0,445277.963173094 445640.109656027,0.0 445640.109656027,0 0)),((111319.490793274 893463.751012645,222638.981586547 893463.751012645,222638.981586547 1006021.06275513,1    11319.490793274 1006021.06275513,111319.490793274 893463.751012645)),((667916.944759641 669141.057044245,667916.944759641 1345708.40840911,1335833.88951928 1345708.40840911,1335833.88951928 669141.057044245,    667916.944759641 669141.057044245),(779236.435552915 781182.214188248,779236.435552915 893463.751012645,890555.926346189 893463.751012645,890555.926346189 781182.214188248,779236.435552915 781182.214188248)))'
g6 = 'GEOMETRYCOLLECTION(LINESTRING(90 190, 120 190, 50 60, 130 10, 190 50, 160 90, 10 150, 90 190), POINT(90 190))'
g7 = 'MULTIPOLYGON (((111319.490793274 111325.142866385,111319.490793274 222684.208505545,222638.981586547 222684.208505545,222638.981586547 111325.142866385,111319.490793274 111325.142866385)),((0 0,111319.490793    274 -111325.142866386,111319.490793274 111325.142866385,-222638.981586547 334111.17140196,0 0)))'
g8 = 'MULTIPOLYGON (((111319.490793274 111325.142866385,111319.490793274 222684.208505545,222638.981586547 222684.208505545,222638.981586547 111325.142866385,111319.490793274 111325.142866385)),((0 0,111319.490793274 -111325.142866386,111319.490793274 111325.142866385,-222638.981586547 334111.17140196,0 0)))'

geo1 = wkt.loads(g1)
geo2 = wkt.loads(g2)
geo3 = wkt.loads(g3)
geo4 = wkt.loads(g4)
geo5 = wkt.loads(g5)
geo6 = wkt.loads(g6)
geo7 = wkt.loads(g7)
# for f in list(geo5):
#     print(f)

# exit(0)

def line_to_points(geom):
    if geom.strip().upper().startswith('LINESTRING'):
        points = geom.split('(')[1].split(')')[0]
        arr = points.split(',')
        xs = ['POINT (%s)' % x.strip() for x in arr]
        return xs

def polygon_to_points(geom):
    if geom.strip().upper().startswith('POLYGON'):
        points = geom.split('((')[1].split('))')[0]
        arr = points.split(',')
        xs = ['POINT (%s)' % x.strip().replace('(', '').replace(')', '') for x in arr]
        return xs

def geometrycollection_tolist(geom):
    if is_geometrycollection(geom):
        gc = wkt.loads(geom)
        return [x.to_wkt() for x in list(gc)]

def point_distance(geox, geoy):
    if is_geometry(geoy) and geox.strip().upper().startswith('POINT'):
        # print(geox, geoy)
        p = wkt.loads(geox)
        g = wkt.loads(geoy)
        # print('============ passed ============')
        return p.distance(g)

def linestring_distance(geox, geoy):
    if is_geometry(geoy) and geox.strip().upper().startswith('LINESTRING'):
        xs = line_to_points(geox)
        distance_arr = [point_distance(x, geoy) for x in xs]
        
        return max(distance_arr)

def polygon_distance(geox, geoy):
    if is_geometry(geoy) and geox.strip().upper().startswith('POLYGON'):
        xs = polygon_to_points(geox)
        distance_arr = [point_distance(x, geoy) for x in xs]
        
        return max(distance_arr)

def geometry_distance(geox, geoy):
    if is_point(geox):
        return point_distance(geox, geoy)
    
    if is_linestring(geox):
        return linestring_distance(geox, geoy)

    if is_polygon(geox):
        return polygon_distance(geox, geoy)

def arc_distance(geox, geoy):
    if is_empty(geox) or is_empty(geoy):
        return 0.0

    if is_geometrycollection(geox) and is_geometrycollection(geoy):
        gcx = geometrycollection_tolist(geox)
        gcy = geometrycollection_tolist(geoy)

        arr = []
        for gx in gcx:
            distance_arr = [geometry_distance(gx, gy) for gy in gcy]
            arr.append(min(distance_arr))
        
        return max(arr)
    
    if is_geometry(geox) and is_geometrycollection(geoy):
        gc = geometrycollection_tolist(geoy)
        distance_arr = [geometry_distance(geox, x) for x in gc]
        
        return max(distance_arr)

    if is_geometry(geoy) and is_geometrycollection(geox):
        return arc_distance(geoy, geox)
    
    if is_geometry(geox) and is_geometry(geoy):
        return geometry_distance(geox, geoy)
    

# print(arc_distance(g7, g8))
# exit(0)


arc_path = '/tmp/arctern_results/run_test_st_transform.csv'
pgs_path = 'expected/results/st_transform.out'

with open(arc_path, 'r') as arc, open(pgs_path, 'r') as pgs:
    arcs = arc.readlines()[1:]
    pgss = pgs.readlines()[1:]
    for x, y in zip(arcs, pgss):
        # print(x)
        print(arc_distance(x, y))
    
