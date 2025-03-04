#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset


NUM_WORKERS=${NUM_WORKERS:-5}
TIMEOUT=${TIMEOUT:-180}

if [ "$IS_LOCAL" = "on" ];
then
    uvicorn main:app --host 0.0.0.0 --port 8070 --log-level=info --reload
else
    uvicorn main:app --host 0.0.0.0 --port 80 --log-level=info
fi

exec "$@"