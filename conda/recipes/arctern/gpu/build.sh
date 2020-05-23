# show environment
printenv

# Cleanup local git
git clean -xdf

# Build libarctern
./ci/scripts/cpp_build.sh clean
./ci/scripts/cpp_build.sh --build_type=Release -g -v

# Build pyarctern
./ci/scripts/python_build.sh clean
./ci/scripts/python_build.sh --install_prefix=$PREFIX --library_dirs=$PREFIX/lib

# Build arctern_pyspark
./ci/scripts/pyspark_build.sh clean
./ci/scripts/pyspark_build.sh --install_prefix=$PREFIX
