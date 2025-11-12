# 1. Task: rp (Rating Prediction)
# Dataset: Beauty
python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/rp/Beauty-reflection.jsonl
python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/rp/Beauty-v2.jsonl --model_path ckpts/xxxx/epoch-0

# 2. Task: sr (Sequential Recommendation)
# Dataset: Beauty
python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/sr/Beauty-reflection.jsonl
python main.py -m RLHFTraining --config_path config/training/ppo-main.json --epochs 1 --data_file data/ppo/sr/Beauty-v1.jsonl --model_path ckpts/xxxx/epoch-0