#!/usr/bin/env bash
# Provision the single EC2 host for the RAGOps Copilot public demo (W7 D3, plan A).
#
# Creates (idempotently where the AWS API allows): an SSH key pair, a security
# group that opens ONLY 22/80/443, and one Ubuntu instance that installs Docker
# on first boot via user-data. Prints the public IP and the exact next steps.
#
# It does NOT deploy the app — provisioning (make the box) and deployment (put
# the app on it) are kept separate so you can re-run one without the other. The
# deploy step is in deploy/README.md.
#
# Prereqs: AWS CLI v2 configured (`aws configure`) with a key that can create
# EC2 + security groups. Everything is parameterised by the env vars below.
#
#   bash deploy/provision-ec2.sh
#
set -euo pipefail

# --- knobs (override by exporting before running) ---------------------------
REGION="${REGION:-us-east-1}"
# t3.medium (4 GB) not t3.small (2 GB): OpenSearch alone locks a 512 MB heap and
# the api container loads torch + the bge embedder into RAM. 2 GB is where the
# OOM killer starts reaping OpenSearch mid-ingest. 4 GB is the safe floor.
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium}"
# 30 GB gp3, not the 8 GB default: the api image carries torch (multi-GB), plus
# the OpenSearch image and the model cache. 8 GB fills during the build.
VOLUME_GB="${VOLUME_GB:-30}"
KEY_NAME="${KEY_NAME:-ragops-key}"
SG_NAME="${SG_NAME:-ragops-sg}"
TAG="${TAG:-ragops-copilot}"
KEY_FILE="$(dirname "$0")/${KEY_NAME}.pem"

# SSH is locked to YOUR current public IP by default (key-only auth is good, but
# not exposing 22 to the whole internet is better). Export SSH_CIDR=0.0.0.0/0 to
# open it wide (e.g. if your IP changes often), accepting the weaker posture.
MY_IP="$(curl -fsS https://checkip.amazonaws.com 2>/dev/null | tr -d '\n' || true)"
SSH_CIDR="${SSH_CIDR:-${MY_IP:+${MY_IP}/32}}"
SSH_CIDR="${SSH_CIDR:-0.0.0.0/0}"

echo "region=$REGION  type=$INSTANCE_TYPE  disk=${VOLUME_GB}GB  ssh_from=$SSH_CIDR"
aws() { command aws --region "$REGION" "$@"; }

# --- 1. SSH key pair --------------------------------------------------------
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
  echo "key pair '$KEY_NAME' already exists (reusing; $KEY_FILE must match it)"
else
  echo "creating key pair '$KEY_NAME' -> $KEY_FILE"
  aws ec2 create-key-pair --key-name "$KEY_NAME" \
    --query 'KeyMaterial' --output text > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
fi

# --- 2. security group: 22 (your IP) + 80 + 443 (world) ---------------------
VPC_ID="$(aws ec2 describe-vpcs --filters Name=is-default,Values=true \
  --query 'Vpcs[0].VpcId' --output text)"
if SG_ID="$(aws ec2 describe-security-groups --filters Name=group-name,Values="$SG_NAME" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)" && [ "$SG_ID" != "None" ]; then
  echo "security group '$SG_NAME' already exists ($SG_ID)"
else
  echo "creating security group '$SG_NAME' in $VPC_ID"
  SG_ID="$(aws ec2 create-security-group --group-name "$SG_NAME" \
    --description "RAGOps Copilot demo: ssh + http + https only" \
    --vpc-id "$VPC_ID" --query 'GroupId' --output text)"
  # Ingress. Egress stays at the default allow-all (the box needs to reach
  # DeepSeek, Let's Encrypt, Docker Hub, HuggingFace).
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
    --ip-permissions \
      "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$SSH_CIDR,Description=ssh}]" \
      "IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0,Description=http-acme}]" \
      "IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges=[{CidrIp=0.0.0.0/0,Description=https}]" \
    >/dev/null
fi

# --- 3. latest Ubuntu 24.04 LTS amd64 AMI (Canonical) -----------------------
AMI_ID="$(aws ec2 describe-images --owners 099720109477 \
  --filters \
    'Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*' \
    'Name=state,Values=available' \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)"
echo "AMI: $AMI_ID"

# --- 4. launch (Docker installs on first boot via user-data) ----------------
USER_DATA="$(base64 -w0 "$(dirname "$0")/bootstrap-instance.sh")"
echo "launching $INSTANCE_TYPE ..."
INSTANCE_ID="$(aws ec2 run-instances \
  --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" --security-group-ids "$SG_ID" \
  --block-device-mappings \
    "DeviceName=/dev/sda1,Ebs={VolumeSize=$VOLUME_GB,VolumeType=gp3}" \
  --user-data "$USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]" \
  --query 'Instances[0].InstanceId' --output text)"
echo "instance $INSTANCE_ID — waiting for it to run ..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

PUBLIC_IP="$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"

cat <<EOF

================================================================================
  EC2 host is up.
    instance : $INSTANCE_ID
    public IP: $PUBLIC_IP
    ssh      : ssh -i $KEY_FILE ubuntu@$PUBLIC_IP

  NEXT (see deploy/README.md):
    1. Point your domain's A record at:  $PUBLIC_IP
       (wait for DNS to propagate: dig +short <your-domain> should return it)
    2. Docker is installing on the box via user-data — give it ~60s. Verify:
         ssh -i $KEY_FILE ubuntu@$PUBLIC_IP 'docker --version && docker compose version'
    3. Ship code + corpus + .env, then bring the stack up (deploy/README.md step 4+).
================================================================================
EOF
