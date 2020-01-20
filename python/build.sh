rm -rf build/
rm -rf dist/
rm -rf zilliz_gis.egg*
python setup.py build
python setup.py install
