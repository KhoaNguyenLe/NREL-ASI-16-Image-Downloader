#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# setup_jetstream.sh — Bootstrap a fresh Jetstream2 VM for SRRL scraping
#
# Run once on a new m3.tiny (or larger) instance:
#   curl -sSL <raw-url> | bash
# Or:
#   scp setup_jetstream.sh exouser@<ip>:~ && ssh exouser@<ip> bash setup_jetstream.sh
# ──────────────────────────────────────────────────────────────
set -euo pipefail

DEPLOY_DIR="$HOME/srrl-pipeline"
LOG_DIR="/var/log/srrl"

echo "=== SRRL Pipeline Setup ==="

# ── 1. System packages ──
echo "[1/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv git logrotate

# ── 2. Create directories ──
echo "[2/7] Creating directories..."
sudo mkdir -p "$LOG_DIR"
sudo chown "$USER:$USER" "$LOG_DIR"
mkdir -p "$DEPLOY_DIR"

# ── 3. Clone or copy pipeline code ──
echo "[3/7] Setting up pipeline code..."
if [ ! -d "$DEPLOY_DIR/resources" ]; then
    echo "  Copy your Scripts/ directory to $DEPLOY_DIR"
    echo "  Example: scp -r Scripts/* exouser@<ip>:$DEPLOY_DIR/"
    echo "  Then re-run this script."
    # If you have a git repo, uncomment:
    # git clone https://github.com/<user>/<repo>.git "$DEPLOY_DIR"
fi

# ── 4. Python dependencies ──
echo "[4/7] Installing Python dependencies..."
python3 -m venv "$DEPLOY_DIR/venv"
source "$DEPLOY_DIR/venv/bin/activate"
pip install --quiet requests huggingface_hub pandas numpy pillow torch torchvision

# ── 5. Environment file ──
echo "[5/7] Setting up environment..."
if [ ! -f "$HOME/.env" ]; then
    cat > "$HOME/.env" << 'ENVEOF'
# HuggingFace write token (get from https://huggingface.co/settings/tokens)
HF_TOKEN=hf_REPLACE_ME

# Data directories (adjust if your volume is mounted elsewhere)
SRRL_IMAGE_DIR=/media/volume/Primary-Dataset
SRRL_METEO_DIR=/media/volume/Primary-Dataset/meteorological
ENVEOF
    echo "  Created ~/.env — EDIT THIS FILE to set your HF_TOKEN"
    echo "  Run: nano ~/.env"
fi

# ── 6. Install systemd units ──
echo "[6/7] Installing systemd services..."

# Update ExecStart paths to use venv python
VENV_PYTHON="$DEPLOY_DIR/venv/bin/python3"

sudo cp "$DEPLOY_DIR/resources/deploy/scraper.service" /etc/systemd/system/
sudo sed -i "s|/usr/bin/python3|$VENV_PYTHON|g" /etc/systemd/system/scraper.service

sudo cp "$DEPLOY_DIR/resources/deploy/daily-sync.service" /etc/systemd/system/
sudo sed -i "s|/usr/bin/python3|$VENV_PYTHON|g" /etc/systemd/system/daily-sync.service

sudo cp "$DEPLOY_DIR/resources/deploy/daily-sync.timer" /etc/systemd/system/

# Logrotate
sudo cp "$DEPLOY_DIR/resources/deploy/logrotate-srrl" /etc/logrotate.d/srrl

sudo systemctl daemon-reload

# ── 7. Enable and start ──
echo "[7/7] Enabling services..."
sudo systemctl enable --now scraper.service
sudo systemctl enable --now daily-sync.timer

echo ""
echo "=== Setup complete ==="
echo ""
echo "Status:"
echo "  systemctl status scraper        # continuous scraper"
echo "  systemctl status daily-sync.timer  # daily upload timer"
echo "  journalctl -u scraper -f        # live scraper logs"
echo "  tail -f /var/log/srrl/scraper.log"
echo ""
echo "IMPORTANT: Edit ~/.env and set your HF_TOKEN before the daily sync runs!"
echo ""
echo "Disk usage:"
echo "  df -h /media/volume/            # check volume space"
echo "  du -sh /media/volume/Primary-Dataset/  # dataset size"
echo ""
echo "To manually trigger a sync:"
echo "  sudo systemctl start daily-sync.service"
