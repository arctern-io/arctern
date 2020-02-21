#!/bin/bash
if [ -d cpp/build ];then
    rm -rf cpp/build
fi
mkdir cpp/build
pushd cpp/build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DBUILD_UNITTEST=OFF  -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=OFF
make install
popd

pushd python
python setup.py build
python setup.py install
popd

pushd spark/pyspark
python setup.py build
python setup.py install
popd
