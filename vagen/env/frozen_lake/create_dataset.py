import re
import os
import json
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import datasets
import multiprocessing as mp
from functools import partial
import numpy as np

from vagen.env.create_dataset import DatasetCreator
from vagen.env.frozen_lake.env import FrozenLakeInterface
from vagen.env.frozen_lake.utils import get_shortest_action_path, plot_animation

class FrozenLakeDatasetCreator(DatasetCreator):
    def _process_seed(self, seed: int, max_action_length: int = 5):
        self.env_config["seed"] = seed
        env_interface = FrozenLakeInterface(self.env_config, self.interface_config)
        env_interface.reset(seed=seed)
        gt_action_sequence = get_shortest_action_path(
            seed=seed, 
            size=env_interface.env.size,
            p=env_interface.env.p,
            is_slippery=env_interface.env.is_slippery, 
            MAX_DEPTH=max_action_length,
        )
        if len(gt_action_sequence) > max_action_length:
            return seed, []
        return seed, gt_action_sequence
    
    def create_filtered_dataset(
        self,
        seed: int = 0,
        train_ratio: float = 0.8,
        max_action_length: int = 5,
        n_candidate: int = 20000,
        force_gen: bool = False,
    ):
        """
        Create a filtered dataset with given seeds
        """
        train_file_path = os.path.join(self.data_dir, 'train.parquet')
        test_file_path = os.path.join(self.data_dir, 'test.parquet')
        
        # Check if files already exist and force_gen is False
        if not force_gen and os.path.exists(train_file_path) and os.path.exists(test_file_path):
            print(f"Dataset files already exist at {self.data_dir}. Skipping generation.")
            print(f"Use --force-gen to override and regenerate the dataset.")
            return

        num_processes = mp.cpu_count()
        print(f"Using {num_processes} processes for seed processing")
        pool = mp.Pool(processes=num_processes)
        process_seed_partial = partial(self._process_seed, max_action_length=max_action_length)
        seeds = range(seed, seed + n_candidate)
        results = list(tqdm(pool.imap(process_seed_partial, seeds), total=len(seeds), desc="Processing seeds"))
        pool.close()
        pool.join()

        valid_seeds = [seed for seed, gt_action_sequence in results if gt_action_sequence and len(gt_action_sequence) <= max_action_length]
        train_size = int(len(valid_seeds) * train_ratio)
        test_size = len(valid_seeds) - train_size
        print(f"Train size: {train_size}, Test size: {test_size}")
        # Analyze statistics of action sequences
        action_lengths = [len(gt_action_sequence) for _, gt_action_sequence in results if gt_action_sequence]
        
        # Calculate basic statistics
        avg_length = np.mean(action_lengths) if action_lengths else 0
        median_length = np.median(action_lengths) if action_lengths else 0
        min_length = min(action_lengths) if action_lengths else 0
        max_length = max(action_lengths) if action_lengths else 0
        
        # Count frequency of each action length
        length_counts = {}
        for length in action_lengths:
            length_counts[length] = length_counts.get(length, 0) + 1
        
        # Calculate percentage of valid seeds
        valid_percentage = (len(valid_seeds) / n_candidate) * 100
        
        # Print statistics
        print("\nAction Sequence Statistics:")
        print(f"Total candidates processed: {n_candidate}")
        print(f"Valid seeds found: {len(valid_seeds)} ({valid_percentage:.2f}%)")
        print(f"Average action length: {avg_length:.2f}")
        print(f"Median action length: {median_length}")
        print(f"Min action length: {min_length}")
        print(f"Max action length: {max_length}")
        print("\nAction length distribution:")
        for length in sorted(length_counts.keys()):
            count = length_counts[length]
            percentage = (count / len(action_lengths)) * 100
            print(f"  Length {length}: {count} instances ({percentage:.2f}%)")

        self.create_dataset(valid_seeds, train_size, test_size, force_gen=force_gen)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--n_candidate', type=int, default=20000)
    parser.add_argument('--max_action_length', type=int, default=None)
    parser.add_argument('--force-gen', action='store_true', 
                        help='Force dataset generation even if files already exist')
    parser.add_argument('--data_dir', type=str, default='data/frozen_lake',)

    parser.add_argument('--size', type=int, default=8,
                    help='Size of the square grid for the environment')
    parser.add_argument('--is_slippery', action='store_true',
                    help='Whether the ice is slippery or not')
    parser.add_argument('--p', type=float, default=0.8,
                    help='Probability of moving in the intended direction')
    parser.add_argument('--desc', type=str, default=None,
                    help='Custom map layout as a string, otherwise a random map will be generated')
    parser.add_argument('--visual_env', action='store_true',
                        help='Whether to use visual environment')
    
    parser.add_argument('--max_action_per_step', type=int, default=1,
                        help='Maximum number of actions per step')
    parser.add_argument('--max_action_penalty', type=float, default=-0.1,
                        help='Penalty for exceeding the maximum number of actions per step')
    parser.add_argument('--format_reward', type=float, default=0.5,
                        help='Reward for correct formatting')
    import os
    if 'PYTHONHASHSEED' not in os.environ:
        os.environ['PYTHONHASHSEED'] = '0'
        print("Set PYTHONHASHSEED to 0 for reproducibility")
    else:
        print(f"PYTHONHASHSEED already set to {os.environ['PYTHONHASHSEED']}")
    

    args = parser.parse_args()
    args.name = 'frozen_lake'
    args.env_config = {
        'desc': args.desc,
        'size': args.size,
        'is_slippery': args.is_slippery,
        'p': args.p,
        'visual_env': args.visual_env
    }
    args.interface_config = {
        'max_action_per_step': args.max_action_per_step,
        'max_action_penalty': args.max_action_penalty,
        'format_reward': args.format_reward,
    }
    creator = FrozenLakeDatasetCreator(config=vars(args))
    if args.max_action_length:
        creator.create_filtered_dataset(
            seed=args.start_seed,
            train_ratio=args.train_ratio,
            max_action_length=args.max_action_length,
            n_candidate=args.n_candidate,
            force_gen=args.force_gen
        )
    else:
        train_size = int(args.train_ratio * args.n_candidate)
        test_size = args.n_candidate - train_size
        creator.create_dataset(
            seed=args.start_seed,
            train_size=train_size,
            test_size=test_size,
            force_gen=args.force_gen)
