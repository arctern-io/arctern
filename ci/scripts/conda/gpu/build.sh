#!/bin/bash

set -ex

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
PROJECT_ROOT_PATH="${SCRIPTS_DIR}/../../../.."

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:$PATH
export PARALLEL_LEVEL=4

# Set versions of packages needed to be grabbed

# Switch to project root; also root of repo checkout
cd $PROJECT_ROOT_PATH

# Get latest tag and number of commits since tag
export GIT_DESCRIBE_TAG=`git describe --abbrev=0 --tags`
export GIT_DESCRIBE_NUMBER=`git rev-list ${GIT_DESCRIBE_TAG}..HEAD --count`

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "nightly" ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Activate conda env..."
eval "$(conda shell.bash hook)"
conda activate arctern

if [ -n "${CONDA_CUSTOM_CHANNEL}" ]; then
    conda config --add channels ${CONDA_CUSTOM_CHANNEL}
    conda config --set show_channel_urls yes
    conda config --show channels
fi

conda install --yes --quiet conda-build anaconda-client -c conda-forge

logger "Check versions..."
python --version
gcc --version
g++ --version
conda list

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

################################################################################
# BUILD - Conda package builds (conda deps: libarctern <- arctern <- arctern-spark)
################################################################################

logger "Build conda pkg for libarctern..."
source ci/scripts/conda/gpu/libarctern/build_libarctern.sh

logger "Build conda pkg for arctern..."
source ci/scripts/conda/gpu/arctern/build_arctern.sh

logger "Build conda pkg for arctern-spark..."
source ci/scripts/conda/gpu/arctern-spark/build_arctern-spark.sh

################################################################################
# UPLOAD - Conda packages
################################################################################

source ci/scripts/conda/gpu/upload_package.sh
source ci/scripts/conda/gpu/upload_anaconda.sh
