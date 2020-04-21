#!/bin/bash

set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPTS_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

REPO_ROOT_PATH="${SCRIPTS_DIR}/../.."

HELP="
Usage:
  $0 [flags] [Arguments]

    -e [CONDA_ENV] or --conda_env=[CONDA_ENV]
                              Setting conda activate environment
    -h or --help              Print help information


Use \"$0  --help\" for more information about a given command.
"

ARGS=`getopt -o "e:h" -l "conda_env::,help" -n "$0" -- "$@"`

eval set -- "${ARGS}"

while true ; do
        case "$1" in
                -e|--conda_env)
                        case "$2" in
                                "") echo "Option conda_env, no argument"; exit 1 ;;
                                *)  CONDA_ENV=$2 ; shift 2 ;;
                        esac ;;
                -h|--help) echo -e "${HELP}" ; exit 0 ;;
                --) shift ; break ;;
                *) echo "Internal error!" ; exit 1 ;;
        esac
done

if [[ -n ${CONDA_ENV} ]]; then
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV}
fi

pushd "${REPO_ROOT_PATH}"

find . -name \*.py \
	-and -not -path ./cpp/\* \
	-and -not -path ./ci/\* \
	-and -not -path ./doc/\* \
	-and -not -path ./gui/client/\* \
	-and -not -path ./docker/\* \
| sed 's/./\\&/g' \
| xargs pylint -j 4 -ry --msg-template='{path}:{line}:{column}: {msg_id}: {msg} ({symbol})' --ignore="" \
"$@" || exit 1

popd
