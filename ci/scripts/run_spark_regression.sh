#!/bin/bash

set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

TESTS_DIR="${SCRIPTS_DIR}/../../tests"

HELP="
Usage:
  $0 [flags] [Arguments] [application-arguments]

    --master [URL]            The master URL for the cluster (e.g. spark://23.195.26.187:7077)
    -h or --help              Print help information

    application-arguments: Arguments passed to the main method of your main class, if any


Use \"$0  --help\" for more information about a given command.
"

ARGS=`getopt -o "h" -l "master:,help" -n "$0" -- "$@"`

eval set -- "${ARGS}"

while true ; do
        case "$1" in
                --master)
                        # o has an optional argument. As we are in quoted mode,
                        # an empty parameter will be generated if its optional
                        # argument is not found.
                        case "$2" in
                                "") echo "Option master, no argument"; exit 1 ;;
                                *)  MASTER_URL=$2 ; shift 2 ;;
                        esac ;;
                -h|--help) echo -e "${HELP}" ; exit 0 ;;
                --) shift ; break ;;
                *) echo "Internal error!" ; exit 1 ;;
        esac
done

# Set defaults for vars modified by flags to this script
MASTER_URL=${MASTER_URL:="spark://127.0.0.1:7077"}

pushd ${TESTS_DIR}

/opt/spark/bin/spark-submit --master ${MASTER_URL} $@
python collect_results.py
python compare.py
python test_vectory_impl.py 

popd
