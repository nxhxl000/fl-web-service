#!/usr/bin/env bash
# One-shot Docker installation on every client VM in nodes.conf.
#
# Each VM gets the official Docker install script + the current SSH user
# added to the `docker` group. Interactive — sudo will prompt for the
# password on each VM (so 10 prompts total, one per VM).
#
# Run once before scripts/deploy_clients.sh.
#
# Usage:
#   scripts/install_docker_clients.sh

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

  echo ""
  echo "── ${NAME} (${HOST}:${PORT}, user ${USER}) ──"

  # ssh -t allocates a TTY so sudo can prompt.
  ssh -t -i "${KEY}" -p "${PORT}" -o StrictHostKeyChecking=accept-new \
      "${USER}@${HOST}" \
      'if command -v docker >/dev/null; then
         echo "  docker already installed: $(docker --version)"
       else
         curl -fsSL https://get.docker.com | sudo sh
         sudo usermod -aG docker $USER
         echo "  docker installed on $(hostname)"
       fi'
done

echo ""
echo "Done. Verify on a VM:"
echo "  ssh ... 'docker --version'"
