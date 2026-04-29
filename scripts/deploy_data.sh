#!/usr/bin/env bash
# Distribute ImageFolder partitions to client VMs over SSH.
#
# Reads nodes.conf for SSH coordinates. Maps NODE_<N> → client_<N-1>:
#   NODE_1=vm-client1   ← partition client_0
#   NODE_2=vm-client2   ← partition client_1
#   ...
#   NODE_10=fl-client4  ← partition client_9
#
# Each VM gets its slice rsynced to ~/fl-data/. The Docker container we
# launch later will mount this folder as /data.
#
# Usage:
#   scripts/deploy_data.sh <imagefolder_root>
#
# Example:
#   scripts/deploy_data.sh data/partitions/cifar100__iid__n10__s42__imagefolder

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <imagefolder_root>" >&2
  exit 2
fi

PARTITION_ROOT="$1"
if [[ ! -d "${PARTITION_ROOT}" ]]; then
  echo "not a directory: ${PARTITION_ROOT}" >&2
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

echo "Distributing ${PARTITION_ROOT} → 10 client VMs"

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

  partition="${PARTITION_ROOT}/client_$((i-1))"
  if [[ ! -d "${partition}" ]]; then
    echo "  skip ${NAME}: partition not found at ${partition}"
    continue
  fi

  echo ""
  echo "── ${NAME} (${HOST}:${PORT}, user ${USER}) ──"
  echo "  partition: ${partition} ($(du -sh "${partition}" | cut -f1))"

  ssh -i "${KEY}" -p "${PORT}" -o StrictHostKeyChecking=accept-new \
      "${USER}@${HOST}" "mkdir -p ~/fl-data"

  rsync -azh --delete --info=progress2 \
      -e "ssh -i ${KEY} -p ${PORT}" \
      "${partition}/" "${USER}@${HOST}:~/fl-data/"
done

echo ""
echo "Done. Verify on a VM with:"
echo "  ssh ... 'ls ~/fl-data | head'"
