# Deploy — RAGOps Copilot public demo (W7 D3)

Plan **A**: one EC2 host runs the whole `docker compose` stack; **Caddy** is the
only thing on the internet (ports 80/443), terminating HTTPS and reverse-proxying
to the API. OpenSearch and the API publish **no** host ports — they live on the
private compose network. Two independent locks keep OpenSearch off the internet:
the security group (only 22/80/443 open) and the compose topology (no published
port).

```
            :443 / :80
  internet ─────────────▶ Caddy ──┬── /            → static frontend (/srv)
  (world)                (TLS,     │
                          only     └── /ask /health → api:8000 ──▶ opensearch:9200
                          public                                    (private net,
                          service)                                   no host port)
```

## Files

| File | Role |
|---|---|
| `provision-ec2.sh` | Create key pair + security group (22/80/443) + Ubuntu instance (Docker via user-data). Prints the public IP. |
| `bootstrap-instance.sh` | First-boot user-data: installs Docker + compose plugin, raises `vm.max_map_count`. |
| `deploy.sh` | rsync repo → box, set `DOMAIN` in the box's `.env`, `compose up`, ingest, smoke-test. Re-run on every code change. |
| `Caddyfile` | Edge config: auto-HTTPS + static host + reverse proxy. |
| `../docker-compose.prod.yml` | Prod topology (OpenSearch + API private, Caddy public). |

## One-time prerequisites (on your laptop)

```bash
# 1. AWS CLI v2 configured with an IAM key that can create EC2 + security groups
aws configure                       # region e.g. us-east-1
aws sts get-caller-identity         # confirm

# 2. You own a domain you can add an A record to (e.g. ragops.example.com)
```

## Deploy

```bash
cd ~/projects/ragops-copilot

# 1. Provision the box (~1 min). Prints the public IP and the ssh command.
bash deploy/provision-ec2.sh
#    Override defaults if needed:
#    REGION=us-west-2 INSTANCE_TYPE=t3.medium bash deploy/provision-ec2.sh

# 2. Point your domain's A record at the printed IP. Wait for propagation:
dig +short ragops.example.com       # should return the EC2 IP

# 3. Ship the app + corpus + secrets and bring the stack up (~a few min: the
#    first image build pulls torch). Ingest runs automatically.
HOST=<public-ip> DOMAIN=ragops.example.com \
  [ACME_EMAIL=you@example.com] bash deploy/deploy.sh
```

`deploy.sh` ships `.env` and `data/chunks.jsonl` (both gitignored — they exist
only on your laptop) up to the box. The DeepSeek key never enters git or an image
layer; it is injected into the api container at run time from the box's `.env`.

## Verify (the D3 done-check)

```bash
curl https://ragops.example.com/health          # {"status":"ok", ...}
curl -X POST https://ragops.example.com/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is PagedAttention?"}'    # cited answer
# open https://ragops.example.com/ in a browser — ask a compare + a refusal question

# OpenSearch must NOT be reachable from the internet — this should hang/refuse:
curl --max-time 5 http://ragops.example.com:9200 ; echo "exit=$?"
```

The first HTTPS request may take a few seconds while Caddy fetches the
certificate. If it fails, the A record probably hasn't propagated yet — Let's
Encrypt validates over port 80 against the domain.

## Operate

```bash
ssh -i deploy/ragops-key.pem ubuntu@<ip>
cd ragops-copilot
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f caddy   # TLS + access logs
docker compose -f docker-compose.prod.yml logs -f api     # per-request telemetry

# Redeploy after a code change (from your laptop):
HOST=<ip> DOMAIN=ragops.example.com bash deploy/deploy.sh
```

## Cost control

`t3.medium` + 30 GB gp3 ≈ **$0.05/hr** (~$35/mo) if left on. **Stop it when
you're not demoing:**

```bash
aws ec2 stop-instances  --instance-ids <id>   # billing for compute pauses
aws ec2 start-instances --instance-ids <id>   # note: public IP changes on start
# Tear everything down for good:
aws ec2 terminate-instances --instance-ids <id>
```

> The public IP changes when you stop/start. For a stable demo URL either keep
> the box running, allocate an Elastic IP, or just re-point the A record after
> each start.
