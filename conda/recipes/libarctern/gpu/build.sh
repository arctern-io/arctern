# show environment
printenv

# Cleanup local git
git clean -xdf

./ci/scripts/cpp_build.sh clean
./ci/scripts/cpp_build.sh --build_type=Release -g -v
