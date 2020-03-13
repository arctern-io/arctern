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

cd "${REPO_ROOT_PATH}"

find . -name \*.py \
	-and -not -path ./cpp/\* \
	-and -not -path ./ci/\* \
	-and -not -path ./doc/\* \
	-and -not -path ./docker/\* \
| sed 's/./\\&/g' \
| xargs pylint -j 4 -ry --msg-template='{path}:{line}:{column}: {msg_id}: {msg} ({symbol})' --ignore="" \
"$@" || exit 1
