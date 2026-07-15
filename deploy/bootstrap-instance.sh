#!/usr/bin/env bash
# EC2 first-boot bootstrap (runs as cloud-init user-data, as root).
#
# Installs Docker Engine + the compose plugin from Docker's official apt repo
# and lets the `ubuntu` user run docker without sudo. That's all a fresh box
# needs before the deploy step (deploy/README.md) can `docker compose up`.
#
# cloud-init runs this once, at first boot. Its output goes to
# /var/log/cloud-init-output.log on the instance — check there if docker is
# missing when you SSH in.
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

# Wait out any apt lock cloud-init's own early updates may still hold.
for _ in $(seq 1 30); do
  fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break
  sleep 2
done

apt-get update -y
apt-get install -y ca-certificates curl gnupg

# Docker's official apt repository (the distro's docker.io lags well behind).
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin

systemctl enable --now docker
usermod -aG docker ubuntu

# OpenSearch needs a raised mmap count or it refuses to start; make it durable.
echo 'vm.max_map_count=262144' > /etc/sysctl.d/99-opensearch.conf
sysctl -p /etc/sysctl.d/99-opensearch.conf

echo "bootstrap complete: $(docker --version)"
