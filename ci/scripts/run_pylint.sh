#!/bin/bash

set -e

cd "$(git rev-parse --show-toplevel)"

# pylint3 was replaced with pylint from Ubuntu 19.10
PYLINT=$(command -v pylint3) || true
if [ -z "$PYLINT" ]; then
    PYLINT=$(command -v pylint)
fi

find . -name \*.py \
	-and -not -path ./cpp/\* \
	-and -not -path ./ci/\* \
	-and -not -path ./doc/\* \
	-and -not -path ./docker/\* \
| sed 's/./\\&/g' \
| xargs "${PYLINT}" -j 4 -ry --msg-template='{path}:{line}:{column}: {msg_id}: {msg} ({symbol})' --ignore="" \
"$@"

