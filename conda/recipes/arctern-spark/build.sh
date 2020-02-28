# show environment
printenv

# Cleanup local git
git clean -xdf

./ci/scripts/pyspark_build.sh clean
./ci/scripts/pyspark_build.sh
