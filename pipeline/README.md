# SRRL Data Pipeline — Architecture & Deployment

## Overview

This pipeline continuously scrapes minute-resolution sky images from NREL's SRRL ASI-16 camera, stores them on a Jetstream2 VM, and syncs them (along with paired BMS meteorological data) to HuggingFace daily.

```
                                    Internet
                                       |
  +------------------------------------+------------------------------------+
  |                                    |                                    |
  v                                    |                                    v
+------------------+                   |                 +------------------+
| NREL MIDC        |  HTTPS (1/min)    |                 | HuggingFace      |
| srrlasi.jpg      |<---------+        |        +------->| datasets repo    |
| data_api.pl      |<------+  |        |        |  +---->| knl2366/NREL_..  |
+------------------+       |  |        |        |  |     +------------------+
                           |  |        |        |  |
                    +------+--+--------+--------+--+------+
                    |      |  |   Jetstream2 VM   |  |    |
                    |      |  |   (m3.tiny)        |  |    |
                    |      |  |                    |  |    |
                    |  +---+--+----+    +----------+--+-+  |
                    |  | scraper.py |    | daily_sync.py |  |
                    |  | (systemd)  |    | (systemd tmr) |  |
                    |  +-----+------+    +------+--------+  |
                    |        |                  |           |
                    |        v                  v           |
                    |  +------------------------------------+
                    |  |  /media/volume/Primary-Dataset     |
                    |  |  (Cinder persistent volume)        |
                    |  |                                    |
                    |  |  YYYY/MM/DD/YYYYMMDD-HHMMSS.jpg   |
                    |  |  meteorological/YYYY/MM/DATE.csv   |
                    |  +------------------------------------+
                    +------------------------------------------+
```

## Components

### 1. Continuous Scraper (`scraper.py`)

**What**: Polls NREL's real-time ASI-16 image endpoint every ~60 seconds, deduplicates via MD5 hash, and saves unique images to the Cinder volume.

**How it runs**: As a systemd service (`scraper.service`) with `Restart=always`. If the process crashes, systemd restarts it within 30 seconds.

**Key behaviors**:
- Skips the nighttime blank placeholder (MD5: `604c77fd179dd033f129c4397a8095eb`)
- Skips duplicate frames (same MD5 as previous image)
- Estimates capture time as `(now - 1 minute).replace(second=33)` based on observed ASI-16 timing
- Saves to `YYYY/MM/DD/YYYYMMDD-HHMMSS.jpg` in MST

**Supersedes**: `insurance.sh` (cron-based watchdog) and `hfscraper.py` (scraper + inline HF upload per image, which is slower and creates excessive HF commits).

### 2. Daily Sync (`daily_sync.py`)

**What**: Runs once daily (03:00 MST / 10:00 UTC) via systemd timer. Performs three operations:

1. **Image upload**: Finds all day directories not yet uploaded to HF, deduplicates (SHA256), and batch-uploads each day as a single HF commit.
2. **Meteo download**: Downloads yesterday's BMS 1-minute data from the NREL MIDC API and saves as a daily CSV.
3. **Meteo upload**: Pushes the meteorological directory to HF.

**Why daily (not per-image)**:
- Batch uploads create clean, atomic HF commits (one per day)
- Reduces HF API calls from ~720/day to ~2/day
- MIDC data for a given day is only complete after midnight
- If the sync fails, it catches up on the next run (idempotent)

**Catch-up mode**: On first run (or after downtime), it automatically finds and uploads all un-uploaded days. Use `--start/--end` flags to explicitly sync a date range.

### 3. Configuration (`config.py`)

Centralizes all constants, paths, and credential access. Credentials are loaded from environment variables (`HF_TOKEN`), never hardcoded. The `.env` file on the VM is read by systemd via `EnvironmentFile`.

### 4. Legacy Scripts

| Script | Status | Notes |
|--------|--------|-------|
| `hfscraper.py` | Deprecated | Uploads per-image (slow, noisy HF history). Use `scraper.py` + `daily_sync.py` instead. |
| `hfuploader.py` | Superseded | Batch uploader. Logic absorbed into `daily_sync.py` with improvements. Still works standalone. |
| `insurance.sh` | Superseded | Cron watchdog. Replaced by systemd `Restart=always`. |
| `download_data.py` | Legacy | Older BMS download script (Joshua Hammond). Uses `wget`, hardcoded year ranges. Use `download_and_upload_meteo.py` instead. |
| `utilities.py` | Legacy | File helpers from older project version. |

## Jetstream2 Implementation Guide

### What is Jetstream2?

[Jetstream2](https://jetstream-cloud.org/) is an NSF-funded cloud computing platform operated by Indiana University. Unlike traditional HPC clusters with batch schedulers (SLURM, PBS), Jetstream2 is a full **Infrastructure-as-a-Service (IaaS)** cloud built on OpenStack. You create and manage your own virtual machines with root access — comparable to AWS EC2 or Google Compute Engine, but free for US researchers through the ACCESS program.

Key facts:
- **OpenStack under the hood**: VM lifecycle, networking, storage, and images are all managed by OpenStack services (Nova, Neutron, Cinder, Glance, Swift).
- **Hardware**: AMD EPYC Milan 7713 CPUs, NVIDIA A100/L40S/H100 GPUs, 14 PB Ceph storage, 100 Gbps to Internet2.
- **Primary cloud at IU Bloomington**, with regional sites at ASU, Cornell, UH, and TACC.
- **Management interfaces**: Exosphere (simplified web UI, recommended), Horizon (full OpenStack dashboard), or the OpenStack CLI. All three control the same resource pool.

### Getting an ACCESS Allocation

Jetstream2 is free to use through [ACCESS](https://access-ci.org/) (formerly XSEDE). You need an allocation of "ACCESS Credits" which convert to Jetstream2 Service Units (SUs) at a 1:1 ratio.

| Tier | Max Credits | Effort Required | Turnaround |
|------|------------|-----------------|------------|
| **Explore** | 400,000 | Short abstract | ~Days |
| **Discover** | 1,500,000 | Brief description | ~1-2 weeks |
| **Accelerate** | 3,000,000 | Detailed proposal | ~3-4 weeks |
| **Maximize** | No fixed limit | Full proposal + review | Months |

**For this project, Explore (400K SU) is more than enough.** Our m3.tiny VM costs 1 SU/hr = 8,760 SU/year, so 400K SU lasts 45+ years. Even with occasional larger instances for data processing, Explore is sufficient.

To apply:
1. Create an ACCESS account at https://access-ci.org/
2. Go to https://allocations.access-ci.org/
3. Submit an Explore request (one paragraph describing the project)
4. Once approved, you can launch VMs on Jetstream2

### Creating the VM

We use **Exosphere** (https://jetstream2.exosphere.app/) — the simplified web interface.

#### Step 1: Launch an Instance

1. Log in to Exosphere with your ACCESS credentials
2. Select the **IU** region (primary cloud, largest resource pool)
3. Click **Create** → **Instance**
4. Choose:
   - **Image**: Ubuntu 22.04 (Featured)
   - **Flavor**: `m3.tiny` (1 vCPU, 3 GB RAM, 20 GB root disk)
   - **Name**: `srrl-scraper` (or similar)
   - **Enable web desktop**: No (we only need SSH)
5. Launch — takes ~2-5 minutes

Exosphere automatically:
- Assigns a **floating IP** (public, routable from the internet)
- Opens SSH (port 22) in the security group
- Creates a default user `exouser` with passwordless sudo
- Installs your SSH key from ACCESS

#### Step 2: Attach a Persistent Volume

The root disk (20 GB) is too small and ephemeral. All data goes on a **Cinder volume**.

1. In Exosphere: **Volumes** → **Create Volume**
   - **Size**: 1000 GB (1 TB — the default allocation)
   - **Name**: `Primary-Dataset`
2. **Attach** the volume to your `srrl-scraper` instance
3. SSH in and format/mount it:

```bash
ssh exouser@<floating-ip>

# Find the device (usually /dev/sdb)
lsblk

# Format (ONLY if new — skip if reattaching an existing volume!)
sudo mkfs.ext4 /dev/sdb

# Mount
sudo mkdir -p /media/volume/Primary-Dataset
sudo mount /dev/sdb /media/volume/Primary-Dataset
sudo chown exouser:exouser /media/volume/Primary-Dataset

# Make persistent across reboots
echo '/dev/sdb /media/volume/Primary-Dataset ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab
```

**Important**: The volume persists independently of the VM. If you delete the VM, the volume (and all your data) survives. You can detach it and reattach to a new VM. This is why we store all images on the volume, not the root disk.

#### Step 3: Verify Networking

Jetstream2 does **not** restrict outbound traffic. The scraper can freely reach NREL's servers:

```bash
# Test connectivity to NREL
curl -sI https://midcdmz.nrel.gov/data/rt/srrlasi.jpg | head -5

# Should return:
# HTTP/2 200
# content-type: image/jpeg
# ...
```

No firewall rules or security group changes are needed for outbound HTTP.

### VM Specifications (What We Use)

| Resource | Value | Notes |
|----------|-------|-------|
| Instance flavor | **m3.tiny** | 1 vCPU, 3 GB RAM — more than enough for a scraper |
| SU cost | 1 SU/hr = **8,760 SU/year** | Trivial within any allocation |
| Storage | **Cinder volume** (1 TB) | Persistent across reboots and VM deletion |
| OS | Ubuntu 22.04 LTS | Default Exosphere image |
| Network | Floating IP + unrestricted outbound | 100 Gbps to Internet2 backbone |

Why m3.tiny: The scraper downloads one ~300 KB JPEG per minute and writes it to disk. This uses negligible CPU, memory, and bandwidth. There is no reason to use a larger instance.

### Storage Budget

| Metric | Value |
|--------|-------|
| Images/day (daytime) | ~720 (12 hrs x 60 min) |
| Avg image size | ~200-500 KB |
| Daily image growth | ~0.3-0.7 GB |
| Annual image growth | ~110-260 GB |
| Meteo CSVs/day | 1 (~100 KB) |
| Years until 1 TB full | ~4-8 years |

When the volume approaches capacity, you can:
- Request a volume expansion (Cinder volumes can be resized up, not down)
- Request additional storage quota via an ACCESS supplement
- Archive older data to the Jetstream2 Object Store (S3-compatible, accessible externally)

### Instance Lifecycle and Billing

Understanding how Jetstream2 bills for VMs:

| VM State | SU Cost | What Happens |
|----------|---------|-------------|
| **Running** | 100% (1 SU/hr for m3.tiny) | Normal operation — scraper is active |
| **Stopped** | 50% | Disk preserved, CPU/RAM released. Scraper stops. |
| **Suspended** | 75% | RAM state frozen to disk. Quick resume but still burns SUs. |
| **Shelved** | **0%** | Disk saved as image snapshot, all resources freed. No cost. |

For a long-running scraper, the VM should always be **Running**. Only shelve it if you need to pause data collection for weeks/months.

**Policies to be aware of:**
- Floating IPs on instances shelved for **90+ days** are reclaimed (you'd get a new IP on unshelve)
- Instances shelved for **1+ year** may be subject to deletion
- Neither applies to a running VM

### Deploying the Pipeline

#### Quick Setup (Existing VM)

If you already have a Jetstream2 VM with the volume mounted:

```bash
# 1. SSH in
ssh exouser@<floating-ip>

# 2. Copy the pipeline code to the VM
# (from your local machine)
scp -r "Scripts/" exouser@<ip>:~/srrl-pipeline/

# 3. Run the automated setup
bash ~/srrl-pipeline/resources/deploy/setup_jetstream.sh

# 4. Set your HuggingFace token
nano ~/.env
# Change: HF_TOKEN=hf_REPLACE_ME
# To:     HF_TOKEN=hf_your_actual_token_here
# Save and exit (Ctrl+X, Y, Enter)

# 5. Restart services to pick up the token
sudo systemctl restart scraper
sudo systemctl restart daily-sync.timer

# 6. Verify everything is running
systemctl status scraper
systemctl status daily-sync.timer
tail -f /var/log/srrl/scraper.log
```

#### What `setup_jetstream.sh` Does

The setup script automates the full deployment in 7 steps:

1. **System packages**: Installs `python3-pip`, `python3-venv`, `git`, `logrotate`
2. **Directories**: Creates `~/srrl-pipeline` and `/var/log/srrl/`
3. **Pipeline code**: Placeholder for cloning from GitHub or manual copy
4. **Python venv**: Creates `~/srrl-pipeline/venv` with all dependencies (`requests`, `huggingface_hub`, `pandas`, `numpy`, `pillow`, `torch`, `torchvision`)
5. **Environment file**: Creates `~/.env` template for `HF_TOKEN` and data directory paths
6. **systemd units**: Copies `scraper.service`, `daily-sync.service`, `daily-sync.timer` to `/etc/systemd/system/`, updates `ExecStart` paths to use the venv Python
7. **Enables services**: Starts the scraper and arms the daily sync timer

#### Manual Setup (Step by Step)

If you prefer to understand each piece or the automated script doesn't fit your setup:

```bash
# ── Python environment ──
sudo apt update && sudo apt install -y python3-pip python3-venv
python3 -m venv ~/srrl-pipeline/venv
source ~/srrl-pipeline/venv/bin/activate
pip install requests huggingface_hub pandas numpy

# ── Environment variables ──
cat > ~/.env << 'EOF'
HF_TOKEN=hf_your_token_here
SRRL_IMAGE_DIR=/media/volume/Primary-Dataset
SRRL_METEO_DIR=/media/volume/Primary-Dataset/meteorological
EOF

# ── Test the scraper manually ──
source ~/.env && export HF_TOKEN SRRL_IMAGE_DIR SRRL_METEO_DIR
cd ~/srrl-pipeline/resources
python3 scraper.py
# Watch for "Success" messages; Ctrl+C to stop

# ── Install as systemd services ──
sudo cp deploy/scraper.service /etc/systemd/system/
sudo cp deploy/daily-sync.service /etc/systemd/system/
sudo cp deploy/daily-sync.timer /etc/systemd/system/

# Edit the service files if your paths differ
sudo nano /etc/systemd/system/scraper.service
# Verify: ExecStart, WorkingDirectory, EnvironmentFile paths

sudo systemctl daemon-reload
sudo systemctl enable --now scraper.service
sudo systemctl enable --now daily-sync.timer

# ── Log rotation ──
sudo cp deploy/logrotate-srrl /etc/logrotate.d/srrl
```

### How systemd Services Work

If you haven't used systemd services before, here's the essential mental model:

**systemd** is Linux's init system — it manages services (daemons) that start at boot and run in the background. We use it instead of `cron + nohup` because:
- It automatically restarts crashed processes (`Restart=always`)
- It handles logging, environment variables, and boot-time startup
- It provides clean status reporting (`systemctl status`)
- It's the standard way to run persistent services on modern Linux

#### The Scraper Service (`scraper.service`)

```ini
[Service]
Type=simple                          # A long-running process
ExecStart=/path/to/python3 scraper.py  # The command to run
Restart=always                       # If it crashes, restart it
RestartSec=30                        # Wait 30s before restarting
EnvironmentFile=-/home/exouser/.env  # Load env vars from this file
```

This replaces the old `insurance.sh` cron watchdog. Instead of checking every 5 minutes whether the process is alive and restarting it, systemd does this natively — and more reliably.

#### The Daily Sync Timer (`daily-sync.timer` + `daily-sync.service`)

systemd timers are the modern replacement for cron jobs. We use a **timer unit** that triggers a **service unit**:

```ini
# daily-sync.timer — WHEN to run
[Timer]
OnCalendar=*-*-* 10:00:00 UTC   # Every day at 10:00 UTC (03:00 MST)
Persistent=true                   # If missed (VM was off), run on next boot
RandomizedDelaySec=300            # Jitter up to 5 minutes to avoid thundering herd
```

```ini
# daily-sync.service — WHAT to run
[Service]
Type=oneshot                         # Runs once and exits (not a daemon)
ExecStart=/path/to/python3 daily_sync.py
TimeoutStartSec=1800                 # Allow up to 30 minutes for upload
```

The `Persistent=true` flag is important: if the VM was stopped/rebooted and missed the 10:00 UTC window, systemd will run the sync immediately on the next boot.

### Common Operations

```bash
# ── Service management ──
systemctl status scraper              # Is the scraper running?
systemctl status daily-sync.timer     # Is the timer armed?
systemctl list-timers                 # When will daily-sync next run?

sudo systemctl start scraper          # Start the scraper
sudo systemctl stop scraper           # Stop the scraper
sudo systemctl restart scraper        # Restart after code changes

sudo systemctl start daily-sync.service   # Force a sync right now

# ── Logs ──
journalctl -u scraper -f              # Live scraper logs
journalctl -u scraper --since "1 hour ago"
journalctl -u daily-sync --since today # Today's sync logs
tail -f /var/log/srrl/scraper.log      # File-based logs
tail -f /var/log/srrl/daily-sync.log

# ── Manual sync ──
cd ~/srrl-pipeline/resources
source ~/srrl-pipeline/venv/bin/activate
source ~/.env && export HF_TOKEN SRRL_IMAGE_DIR SRRL_METEO_DIR

python3 daily_sync.py --dry-run                            # Preview
python3 daily_sync.py                                      # Sync yesterday
python3 daily_sync.py --start 2025-01-01 --end 2025-12-31  # Backfill
python3 daily_sync.py --skip-images                        # Only sync meteo
python3 daily_sync.py --skip-meteo                         # Only sync images

# ── Disk usage ──
df -h /media/volume/                  # Volume capacity
du -sh /media/volume/Primary-Dataset/ # Dataset size
du -sh /media/volume/Primary-Dataset/2026/03/  # This month's data
ls /media/volume/Primary-Dataset/$(date +%Y)/$(date +%m)/$(date +%d)/ | wc -l  # Today's images

# ── Health check ──
stat /var/log/srrl/scraper.log | grep Modify  # When was the log last written?
ls -lt /media/volume/Primary-Dataset/$(date +%Y)/$(date +%m)/$(date +%d)/ | head -3
```

### Monitoring and Alerting

#### Basic Health Check

The scraper prints "Success" for each saved image. During daytime (roughly 06:00-18:00 MST in summer, 07:00-17:00 in winter), you should see a new image every ~60 seconds. If the log hasn't been updated in >5 minutes during daytime, something is wrong.

```bash
# Quick health check script — add to crontab if desired
#!/bin/bash
LOG=/var/log/srrl/scraper.log
THRESHOLD=300  # 5 minutes in seconds
LAST_MOD=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
NOW=$(date +%s)
AGE=$((NOW - LAST_MOD))
HOUR=$(date -u +%H)  # UTC hour

# Only alert during NREL daytime (13:00-01:00 UTC = 06:00-18:00 MST)
if [ "$HOUR" -ge 13 ] || [ "$HOUR" -le 1 ]; then
    if [ "$AGE" -gt "$THRESHOLD" ]; then
        echo "ALERT: Scraper log stale for ${AGE}s (threshold: ${THRESHOLD}s)"
        systemctl status scraper
    fi
fi
```

#### What Can Go Wrong

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Scraper not running | Process crashed, systemd gave up | `sudo systemctl restart scraper` and check `journalctl -u scraper` |
| "Network error" in logs | NREL server down or network issue | Usually transient; scraper retries automatically |
| No new images at night | Normal — NREL serves a blank placeholder at night | The scraper correctly skips these |
| Disk full | Volume at capacity | Check with `df -h`; expand volume or archive old data |
| Daily sync failed | HF API error or token expired | Check `journalctl -u daily-sync`; verify `HF_TOKEN` in `~/.env` |
| VM lost floating IP | VM was shelved >90 days | Assign a new floating IP in Exosphere |

### Recovering from Downtime

If the VM was stopped or the scraper was down for a period:

1. **Images from the gap are lost** — the NREL endpoint only serves the *current* image, not historical ones. There is no way to retroactively fill image gaps.
2. **Meteorological data can be backfilled** — BMS data is available historically from the MIDC API:
   ```bash
   python3 daily_sync.py --start 2025-06-01 --end 2025-06-30 --skip-images
   ```
3. **Pending image uploads catch up automatically** — `daily_sync.py` finds all un-uploaded days and syncs them on the next run.

This is why keeping the scraper running continuously is critical. The systemd `Restart=always` directive and the daily sync's `Persistent=true` timer minimize gaps, but extended VM outages will cause permanent image loss.

### Migrating to a New VM

If you need to move to a new Jetstream2 instance (e.g., allocation change, region move):

1. **Detach the Cinder volume** from the old VM (in Exosphere or Horizon)
2. **Create a new VM** (m3.tiny, Ubuntu 22.04)
3. **Attach the volume** to the new VM and mount it at `/media/volume/Primary-Dataset`
4. **Copy the pipeline code** and run `setup_jetstream.sh`
5. **Verify** the scraper resumes and daily sync catches up

The volume contains all the data. The VM is stateless (code + services only) and can be rebuilt in minutes.

### Jetstream2 Object Store (Optional, for External Access)

Jetstream2 includes an **S3-compatible object store** (Swift) accessible from outside the cloud. This could be used as a secondary backup or for sharing data without HuggingFace:

```bash
# Generate S3 credentials (one-time)
openstack ec2 credentials create

# Configure AWS CLI
aws configure --profile js2
# Access Key: <from above>
# Secret Key: <from above>
# Region: <leave blank>

# Upload a backup
aws --profile js2 --endpoint-url https://js2.jetstream-cloud.org:8001 \
    s3 sync /media/volume/Primary-Dataset/ s3://srrl-sky-images/
```

This is not currently part of the pipeline but could be added as an additional backup target.

### Maintenance Checklist

- [ ] **Weekly**: Check scraper is running (`systemctl status scraper`)
- [ ] **Weekly**: Check disk usage (`df -h /media/volume/`)
- [ ] **Monthly**: Verify HF uploads match local data (`python daily_sync.py --dry-run`)
- [ ] **Monthly**: Apply OS security updates (`sudo apt update && sudo apt upgrade`)
- [ ] **Quarterly**: Create a HF dataset snapshot tag (`huggingface_hub.create_tag()`)
- [ ] **Annually**: Renew ACCESS allocation if needed

## HuggingFace Best Practices Applied

1. **Atomic daily commits**: Each day's images are uploaded in a single commit, creating clean git history.
2. **SHA256 dedup**: Prevents duplicate images from inflating repo size.
3. **Directory structure**: `YYYY/MM/DD/` keeps each folder well under HF's 10K file limit.
4. **Paired meteorological data**: BMS CSVs in `meteorological/YYYY/MM/YYYYMMDD.csv` alongside images.
5. **Environment-based auth**: `HF_TOKEN` from env var, never hardcoded.
6. **Snapshot versioning**: Create quarterly HF tags (e.g., `snapshot-2025Q4`) so researchers can pin experiments to a specific dataset version.
7. **Append-only**: Never modify or delete historical data — only add new days.
8. **Dataset card**: `HF_README.md` follows HuggingFace dataset card standards.
