#!/bin/bash
killall mpv 2>/dev/null
sleep 0.3
cd "$(dirname "$0")"
exec mpv --vo=gpu \
  --no-config \
  --volume=10 \
  --script=main.lua \
  --start=300 \
  "${1:-/Users/tobi/src/tries/2026-02-20-motec-parser/sessions/sebring-2026/CT2/26IMSAT02_SEB_CT2_Run03_DHH.MOV}"
