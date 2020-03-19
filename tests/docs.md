# remove a conda env
conda env remove -n arctern

# install conda env for arctern test: env name is 'arctern'
conda create -n arctern -c arctern-dev -c conda-forge arctern-spark

# install shapely
conda install -y shapely