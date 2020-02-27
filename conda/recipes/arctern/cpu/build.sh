# show environment
printenv

# git clean -xdf

./ci/scritps/cpp_build.sh clean
./ci/scritps/cpp_build.sh --build_type Release -l -v
./ci/scritps/python_build.sh clean
./ci/scritps/python_build.sh
