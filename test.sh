#!/bin/bash
killall mpv 2>/dev/null
sleep 0.3
cd "$(dirname "$0")"
exec mpv --vo=gpu \
  --no-config \
  --script=telemetry.lua \
  --start=300 \
  "/Volumes/home/Racing/Incoming/TDS Racing IMSA videos/26R02_SEBRING/D3_RACE/26IMSAR02_SEB_R_Run01_TL.MOV"
