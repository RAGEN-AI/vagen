## Overview
This repo aims for Multi-turn Reinforcement Learning for Vision Language Models.

## Installation

```bash
conda create -n vagen python=3.11 -y
conda activate vagen
git clone git@github.com:JamesKrW/verl.git
cd verl
pip install -e .
cd ../
git clone git@github.com:RAGEN-AI/vagen.git
cd vagen
bash scripts/install.sh
```

## Supported Environments

### 1. Sokoban
Run one of the following scripts:
```bash
bash vagen/examples/sokoban/debug_qwen0_5_1_gpu_grpo.sh
bash vagen/examples/sokoban/debug_qwen0_5_4_gpu_ppo.sh
bash vagen/examples/sokoban/debug_qwen2_5_vl_4gpu_grpo.sh
```

### 2. FrozenLake
Additional dependencies:
```bash
pip install gymnasium
pip install "gymnasium[toy-text]"
```

Run:
```bash
bash vagen/examples/frozen_lake/frozen_debug_qwen2_5vl_4gpu_grpo.sh
```

### 3. EB-Navigation
Additional dependencies:
```bash
pip install ai2thor
pip install numpy==1.25.1  # Update numpy to be compatible with vllm library
```

For headless servers, additional setup is required:
```bash
# Install required packages
apt-get install -y pciutils
apt-get install -y xorg xserver-xorg-core xserver-xorg-video-dummy

# Start X server in a tmux window
python vagen/env/navigation/startx.py 1

# In another terminal, run:
bash vagen/examples/navigation/navi_debug_qwen2_5_vl_4gpu_grpo.sh
```