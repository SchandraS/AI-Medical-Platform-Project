#!/bin/sh
# Renders the nginx proxy target from BACKEND_HOST/BACKEND_PORT before nginx
# starts (nginx itself cannot read environment variables directly). Defaults
# match docker-compose's service name/port, so a plain `docker build` +
# `docker run` with no env vars set still produces a config identical to the
# old static nginx.conf -- local docker-compose behavior is unchanged.
set -e

: "${BACKEND_SCHEME:=http}"
: "${BACKEND_HOST:=backend}"
: "${BACKEND_PORT:=8000}"
export BACKEND_SCHEME BACKEND_HOST BACKEND_PORT

envsubst '${BACKEND_SCHEME} ${BACKEND_HOST} ${BACKEND_PORT}' \
  < /etc/nginx/conf.d/default.conf.template \
  > /etc/nginx/conf.d/default.conf

exec "$@"
