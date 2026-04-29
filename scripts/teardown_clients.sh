#!/usr/bin/env bash
# Stop fl-client container on every VM in nodes.conf. Idempotent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NODES_CONF="${REPO_ROOT}/nodes.conf"
if [[ ! -f "${NODES_CONF}" ]]; then
  echo "nodes.conf not found at ${NODES_CONF}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${NODES_CONF}"

for i in $(seq 1 10); do
  name_var="NODE_${i}_NAME"
  host_var="NODE_${i}_HOST"
  port_var="NODE_${i}_PORT"
  user_var="NODE_${i}_USER"
  key_var="NODE_${i}_KEY"

  NAME="${!name_var}"
  HOST="${!host_var}"
  PORT="${!port_var}"
  USER="${!user_var}"
  KEY="${!key_var/#\~/$HOME}"

  echo "── ${NAME} ──"
  ssh -i "${KEY}" -p "${PORT}" -o StrictHostKeyChecking=accept-new \
      "${USER}@${HOST}" \
      "docker stop fl-client 2>/dev/null && docker rm fl-client 2>/dev/null && echo '  stopped' || echo '  not running'"
done
