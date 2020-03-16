import random
import os
import inspect
import sys
import shutil

curves = ['CIRCULARSTRING', 'COMPOUNDCURVE', 'MULTICURVE', 'CURVEPOLYGON', 'MULTISURFACE']

def to_disk(file_path, content):

    with open(file_path, 'w') as f:
        if isinstance(content, list):
            for line in content:
                f.writelines(line)


def ilnormal(geom):
    geom = geom.strip().upper()
    for x in curves:
        if geom.startswith(x):
            return True
        else:
            continue
    
    return False


def split_data(file_path):
    wholename = os.path.abspath(file_path)
    basename = os.path.basename(file_path)
    basedir = os.path.split(wholename)[0]
    file_name = os.path.splitext(basename)[0]
    file_ext = os.path.splitext(basename)[1]

    normal_arr = []
    ilnormail_arr = []
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]
        ds = [x.strip().upper().split('|') for x in lines]
        for line in ds:
            if len(line) == 1:
                if ilnormal(line[0].strip()):
                    ilnormail_arr.append(line[0].strip() + '\n')
                else:
                    normal_arr.append(line[0].strip() + '\n')
            elif len(line) == 2:
                if ilnormal(line[0].strip()) or ilnormal(line[1].strip()):
                    ilnormail_arr.append(line[0].strip() + '|' + line[1].strip() + '\n')
                else:
                    normal_arr.append(line[0].strip() + '|' + line[1].strip() + '\n')

    if len(normal_arr) > 0:
        if '|' in normal_arr[0]:
            normal_arr.insert(0, 'left|right\n')
        else:
            normal_arr.insert(0, 'geos\n')
        # to_disk(os.path.join(basedir, file_name + '_normal' + file_ext), normal_arr)
        to_disk(os.path.join(basedir, basename), normal_arr)

    if len(ilnormail_arr) > 0:
        if '|' in ilnormail_arr[0]:
            ilnormail_arr.insert(0, 'left|right\n')
        else:
            ilnormail_arr.insert(0, 'geos\n')
        to_disk(os.path.join(basedir, file_name + '_ilnormal' + file_ext), ilnormail_arr)
       

def split_all(folder):
    for f in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, f)):
            print(f)
            if f.endswith('.csv'):
                if not ('_normal' in f or '_ilnormal' in f):
                    split_data(os.path.join(folder, f))
                

if __name__ == '__main__':
    # split_data('./data/area.csv')
    # split_data('./data/crosses.csv')
    # split_all('./data')
    from osgeo import ogr
    from ogr import *
    
    geo1 = 'Point(1 1)'
    geo2 = '''GEOMETRYCOLLECTION(POINT(0 0),POINT(4 5),LINESTRING(3 4.66666666666667,3 3.41421356237309),LINESTRING(2 3,0.75 3),LINESTRING(0.284816755726683 1.13926702290673,0.324866484100664 1.19673014585673,0.3765617228    47253 1.26364587291387,0.431478091517409 1.32794443190811,0.4894832917689 1.38947092197895,0.5 1.4),LINESTRING(0.111111111111111 0.444444444444444,1 0,1.04906767432742 0.00120454379482759,1.09801714032956 0.    00481527332780318,1.14673047445536 0.0108234900352191,1.19509032201613 0.0192147195967696,1.24298017990326 0.0299687468054561,1.29028467725446 0.0430596642677913,1.33688985339222 0.0584559348169794,1.3826834    3236509 0.0761204674887135,1.42755509343028 0.0960107068765569,1.471396736826 0.118078735651645,1.51410274419322 0.142271389999728,1.5555702330196 0.168530387697455,1.59569930449243 0.196792468519356,1.63439    328416365 0.226989546637264,1.67155895484702 0.259048874645041,1.70710678118655 0.292893218813453,1.74095112535496 0.328441045152982,1.77301045336274 0.365606715836355,1.80320753148065 0.404300695507568,1.83    146961230255 0.444429766980399,1.85772861000027 0.485897255806779,1.88192126434836 0.528603263174003,1.90398929312344 0.572444906569719,1.92387953251129 0.617316567634911,1.94154406518302 0.663110146607781,1    .95694033573221 0.709715322745539,1.97003125319454 0.757019820096737,1.98078528040323 0.804909677983873,1.98917650996478 0.853269525544639,1.9951847266722 0.901982859670441,1.99879545620517 0.950932325672583    ,2 1),LINESTRING(1.60020521351513 1.79969108556562,1.59569930449243 1.80320753148065,1.5555702330196 1.83146961230255,1.51410274419322 1.85772861000027,1.471396736826 1.88192126434836,1.42755509343028 1.9039    8929312344,1.38268343236509 1.92387953251129,1.33688985339222 1.94154406518302,1.29028467725446 1.95694033573221,1.24298017990326 1.97003125319454,1.19509032201613 1.98078528040323,1.14673047445536 1.9891765    0996478,1.09801714032956 1.9951847266722,1.04906767432742 1.99879545620517,1 2,0.5 2))'''
    geom1 = ogr.CreateGeometryFromWkt(geo1)
    geom2 = ogr.CreateGeometryFromWkt(geo2)
    gc = ogr.GT_GetCollection(geom2)
    print(gc)
    #for f in dir(geom2):
    #    print(f)
    print(geom1.Equals(geom2))
    print(geom1.Equal(geom2))
    # print(dir(geom2))
    
