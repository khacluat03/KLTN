#! /bin/bash

# Evaluate on Amazon-Beauty

## Evaluate Manager + Analyst
python main.py --main Evaluate --data_file data/Beauty/test.csv --system collaboration --system_config config/systems/collaboration/analyse.json --task rp --steps 1 --max_his 3