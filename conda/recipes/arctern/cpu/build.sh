# show environment
printenv

# Cleanup local git
git clean -xdf

# Build libarctern
./ci/scripts/cpp_build.sh clean
./ci/scripts/cpp_build.sh --build_type=Release -v

# Build pyarctern
./ci/scripts/python_build.sh clean
./ci/scripts/python_build.sh

# Build arctern_pyspark
./ci/scripts/pyspark_build.sh clean
./ci/scripts/pyspark_build.sh
