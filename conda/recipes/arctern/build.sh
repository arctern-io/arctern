# show environment
printenv

# Cleanup local git
git clean -xdf

./ci/scripts/python_build.sh clean
./ci/scripts/python_build.sh
