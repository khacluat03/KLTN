#! /bin/bash

# Feedback generation on Amazon-Beauty
## rating prediction task
### reflection reward
# python main.py --main Feedback --data_file data/Beauty/train.csv --system reflection --system_config config/systems/reflection/config_open.json --task rp --feedback_file data/ppo/rp/Beauty-reflection.jsonl --reward_version reflection
### update reward for reason reward
# python main.py --main RewardUpdate --data_file data/ppo/rp/Beauty-reflection.jsonl --output_file data/ppo/rp/Beauty-v2.jsonl --reward_version v2
## sequential recommendation task
python main.py --main Feedback --data_file data/Beauty/train.csv --system reflection --system_config config/systems/reflection/config_open.json --task sr --max_his 5 --feedback_file data/ppo/sr/Beauty-reflection.jsonl --reward_version reflection
### update reward for reason reward
python main.py --main RewardUpdate --data_file data/ppo/sr/Beauty-reflection.jsonl --output_file data/ppo/sr/Beauty-v1.jsonl --reward_version v1 --task sr
