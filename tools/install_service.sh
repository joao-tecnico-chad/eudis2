#!/bin/bash
# Install Drone Guardian as a systemd service that starts on boot.
#
# Usage:
#   sudo bash tools/install_service.sh
#
# After install:
#   sudo systemctl status drone-guardian    # check status
#   sudo journalctl -u drone-guardian -f    # view logs
#   sudo systemctl restart drone-guardian   # restart
#   sudo systemctl stop drone-guardian      # stop

set -e

USER="guardian"
WORKDIR="/home/$USER/eudis2"
VENV="$WORKDIR/venv/bin/python"
MODEL="models/yolov6n_640.rvc2.tar.xz"

echo "=== Installing Drone Guardian Service ==="

# Create pigpiod service if not exists
if ! systemctl list-unit-files | grep -q pigpiod; then
    echo "Creating pigpiod service..."
    sudo tee /etc/systemd/system/pigpiod.service > /dev/null << EOF2
[Unit]
Description=pigpio daemon
After=local-fs.target

[Service]
Type=forking
ExecStart=/usr/local/bin/pigpiod
ExecStop=/bin/kill -SIGTERM \$MAINPID

[Install]
WantedBy=multi-user.target
EOF2
fi

# Find pigpiod binary
PIGPIO_BIN=$(which pigpiod 2>/dev/null || echo "/usr/local/bin/pigpiod")
if [ -f "$PIGPIO_BIN" ]; then
    # Update ExecStart path
    sudo sed -i "s|ExecStart=.*pigpiod|ExecStart=$PIGPIO_BIN|" /etc/systemd/system/pigpiod.service 2>/dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable pigpiod
    sudo systemctl start pigpiod 2>/dev/null || true
    echo "pigpiod enabled"
else
    echo "WARNING: pigpiod not found. Run: pip install pigpio"
fi

# Create service
echo "Creating drone-guardian service..."
sudo tee /etc/systemd/system/drone-guardian.service > /dev/null << EOF
[Unit]
Description=Drone Guardian — Drone Detection & Neutralisation
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORKDIR
ExecStartPre=/bin/bash -c '$PIGPIO_BIN 2>/dev/null || true'
ExecStart=$VENV tools/detect_and_fire.py --model $MODEL
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable drone-guardian
sudo systemctl start drone-guardian

echo ""
echo "=== Drone Guardian installed ==="
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8080"
echo ""
echo "Commands:"
echo "  sudo systemctl status drone-guardian     # status"
echo "  sudo journalctl -u drone-guardian -f     # logs"
echo "  sudo systemctl restart drone-guardian    # restart"
echo "  sudo systemctl stop drone-guardian       # stop"
echo "  sudo systemctl disable drone-guardian    # disable autostart"
