#!/usr/bin/env bash

python3 code.py
ffmpeg -r 50 -i img/plot_z_%03d.png -c:v libx264 -vf fps=50 -pix_fmt yuv420p bao.mp4
