#!/usr/bin/env bash
# Launch the fl-client container on each VM in nodes.conf.
#
# Stops any existing container, then `docker run -d` with the matching token
# from the tokens file. Each VM mounts ~/fl-data into /data (already
# distributed by scripts/deploy_data.sh).
#
# Usage:
#   scripts/deploy_clients.sh <tokens_file>
#
# Tokens file format (sourced as bash, kept out of git):
#   TOKEN_1=flwc_...
#   TOKEN_2=flwc_...
#   ...
#   TOKEN_10=flwc_...

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <tokens_file>" >&2
  exit 2
fi

TOKENS_FILE="$1"
if [[ ! -f "${TOKENS_FILE}" ]]; then
  echo "tokens file not found: ${TOKENS_FILE}" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NODES_CONF="${REPO_ROOT}/nodes.conf"
if [[ ! -f "${NODES_CONF}" ]]; then
  echo "nodes.conf not found at ${NODES_CONF}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${NODES_CONF}"
# shellcheck disable=SC1090
source "${TOKENS_FILE}"

SERVER_URL="${PUBLIC_SERVER_URL:-http://${SERVER_HOST}/api}"
SUPERLINK_ADDR="${PUBLIC_SUPERLINK_ADDR:-${SERVER_HOST}:${FLOWER_FLEET_PORT}}"
IMAGE="${FL_CLIENT_IMAGE:-ghcr.io/nxhxl000/fl-client:latest}"

echo "Server URL:      ${SERVER_URL}"
echo "SuperLink fleet: ${SUPERLINK_ADDR}"
echo "Image:           ${IMAGE}"

for i in $(seq 1 10); do
  name_var="NODE_${i}_NAME"
  host_var="NODE_${i}_HOST"
  port_var="NODE_${i}_PORT"
  user_var="NODE_${i}_USER"
  key_var="NODE_${i}_KEY"
  token_var="TOKEN_${i}"

  NAME="${!name_var}"
  HOST="${!host_var}"
  PORT="${!port_var}"
  USER="${!user_var}"
  KEY="${!key_var/#\~/$HOME}"
  TOKEN="${!token_var:-}"

  if [[ -z "${TOKEN}" ]]; then
    echo ""
    echo "── ${NAME} — skip (no TOKEN_${i} in tokens file) ──"
    continue
  fi

  echo ""
  echo "── ${NAME} (${HOST}:${PORT}) ──"

  ssh -i "${KEY}" -p "${PORT}" -o StrictHostKeyChecking=accept-new \
      "${USER}@${HOST}" "bash -s" <<REMOTE
set -e
docker stop fl-client 2>/dev/null || true
docker rm fl-client 2>/dev/null || true
docker pull ${IMAGE} >/dev/null
docker run -d --rm --name fl-client \\
  -e FL_TOKEN=${TOKEN} \\
  -e FL_SERVER_URL=${SERVER_URL} \\
  -e FL_SUPERLINK=${SUPERLINK_ADDR} \\
  -v \$HOME/fl-data:/data \\
  ${IMAGE}
echo "  started: \$(docker ps --filter name=fl-client --format '{{.ID}} {{.Status}}')"
REMOTE
done

echo ""
echo "Done. Watch logs on a VM:"
echo "  ssh ... 'docker logs -f fl-client'"
echo ""
echo "Stop everywhere:"
echo "  scripts/teardown_clients.sh"
