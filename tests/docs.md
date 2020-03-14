# remove a conda env
conda env remove -n zgis_dev

# install conda env for arctern test: env name is 'zgis_dev'
conda create -n zgis_dev -c arctern-dev -c conda-forge arctern-spark

# install shapely
conda install -y shapely