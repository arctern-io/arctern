#!/bin/bash

rm -rf libzillizengine.so
rm -rf libzillizengine.a

g++ -fPIC -shared -o libzillizengine.so make_point.cpp -I/home/ljq/czs_3rd/gis_build/third_party/include -I/home/ljq/anaconda2/envs/spark/lib/python3.5/site-packages/pyarrow/include -L/home/ljq/czs_3rd/gis_build/third_party/lib -L/home/ljq/anaconda2/envs/spark/lib/python3.5/site-packages/pyarrow -lgdal -lproj -larrow -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0


# g++ -fPIC -c -o make_point.o make_point.cpp -I/home/ljq/czs_3rd/gis_build/third_party/include -I/home/ljq/anaconda2/envs/spark/lib/python3.5/site-packages/pyarrow/include -L/home/ljq/czs_3rd/gis_build/third_party/lib -L/home/ljq/anaconda2/envs/spark/lib/python3.5/site-packages/pyarrow -lgdal -lproj -larrow -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0
#
# ar rcs libzillizengine.a make_point.o

rm -rf make_point.o

