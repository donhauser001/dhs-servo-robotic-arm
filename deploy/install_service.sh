#!/usr/bin/env bash
# 在 Pi 上把 arm-web.service 装到 systemd 并开机自启。
# 用法：在仓库根执行 `bash deploy/install_service.sh`。
set -euo pipefail

SERVICE_NAME="arm-web"
SRC="$(dirname "$(readlink -f "$0")")/${SERVICE_NAME}.service"
DST="/etc/systemd/system/${SERVICE_NAME}.service"

if [[ ! -f "${SRC}" ]]; then
  echo "找不到 ${SRC}" >&2
  exit 1
fi

echo "[1/4] 复制 ${SRC} → ${DST}"
sudo install -m 0644 "${SRC}" "${DST}"

echo "[2/4] systemctl daemon-reload"
sudo systemctl daemon-reload

echo "[3/4] systemctl enable ${SERVICE_NAME}（开机自启）"
sudo systemctl enable "${SERVICE_NAME}"

echo "[4/4] 重启服务（如已在跑则替换；之前手动跑的 python 进程会被 kill）"
# 先把任何手动起的实例干掉，避免端口占用
pkill -f "webapp.server" 2>/dev/null || true
sleep 1
sudo systemctl restart "${SERVICE_NAME}"

echo
echo "─── systemctl status ───"
sudo systemctl --no-pager --full status "${SERVICE_NAME}" || true
echo
echo "完成。常用命令："
echo "  systemctl status arm-web      # 看状态"
echo "  journalctl -u arm-web -f      # 实时日志"
echo "  sudo systemctl restart arm-web"
echo "  sudo systemctl stop arm-web"
echo "  sudo systemctl disable arm-web  # 取消开机自启"
