## Overview
This repo aims for Multi-turn Reinforcement Learning for Vision Language Models.

## Installation

```
conda create -n vagen python=3.10 -y
git clone git@github.com:JamesKrW/verl.git
cd verl
pip install -e .
cd ../
git clone git@github.com:RAGEN-AI/vagen.git
cd vagen
bash scripts/install.sh
```

## Run
```
bash vagen/examples/sokoban/debug_qwen0_5_1_gpu_grpo.sh
bash vagen/examples/sokoban/debug_qwen0_5_4_gpu_ppo.sh
bash vagen/examples/sokoban/debug_qwen2_5_vl_4gpu_grpo.sh
```
