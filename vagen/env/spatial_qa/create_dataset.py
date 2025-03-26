"""
Preprocess dataset for genereal tasks
"""

import os
import json
from datasets import Dataset
import argparse
from typing_extensions import override

from vagen.env.create_dataset import DatasetCreator

# TODO: now env_config is set to be absolute path of the data file

class SpatialQADatasetCreator(DatasetCreator):

    @override
    def create_dataset(
        self,
        data_file: str,
        train_ratio: float = 0.8,
        force_gen: bool = False,
        split: str = 'train',
    ):
        """
        Create a dataset in parquet format
        Args:
            data_file: path to the data file
            train_ratio: ratio of train dataset
            force_gen: force dataset generation even if files already exist
            split: split to create

        - env_config is set to be absolute path of the data file
        - seed is set to be the index of the instance
        - type is the type of the QA corresponding to the instance indexed by seed
        """
        train_file = os.path.join(self.data_dir, 'train.parquet')
        test_file = os.path.join(self.data_dir, 'test.parquet')
        with open(data_file, 'r') as f:
            QAs = json.load(f)['QAs']
        
        # Check if files already exist and force_gen is False
        if not force_gen and os.path.exists(train_file) and os.path.exists(test_file):
            print(f"Dataset files already exist at {self.data_dir}. Skipping generation.")
            print(f"Use --force-gen to override and regenerate the dataset.")
            return
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        def _create_instance(idx):
            # Create env_config with type and QAs for this type
            qa_dict = QAs[idx]
            qa_type = qa_dict['type']
            env_config = {
                'type': qa_type,
                'data_path': os.path.abspath(data_file)
            }
            
            env_settings = {
                'env_name': self.env_name,
                'env_config': env_config,
                'interface_config': self.interface_config,
                'seed': idx
            }

            # TODO: no reward model defined here for the reward will be generated while rollout
            return {
                "data_source": self.env_name,
                "prompt": [{"role": "user", "content": ''}],
                "extra_info": {"split": split, **env_settings}
            }
        
        
        # Calculate split indices based on train_ratio
        num_total = len(QAs)
        num_train = int(num_total * train_ratio)
        
        # Create instances for train and test splits
        train_instances = [_create_instance(i) for i in range(num_train)]
        test_instances = [_create_instance(i) for i in range(num_train, num_total)]
        
        train_dataset = Dataset.from_list(train_instances)
        test_dataset = Dataset.from_list(test_instances)

        def make_map_fn(split):
            def process_fn(example, idx):
                return example
            return process_fn
        
        train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

        train_dataset.to_parquet(train_file)
        test_dataset.to_parquet(test_file)
        print(f"Dataset successfully generated at {self.data_dir}")

        







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-gen', action='store_true', 
                        help='Force dataset generation even if files already exist')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of train dataset')
    parser.add_argument('--data_dir', type=str, default='data/spatial_qa',)

    # env_config
    parser.add_argument('--qa_data_file', type=str, default='data/spatial_qa/spatial_qa.json',)
    
    # interface_config
    parser.add_argument('--max_action_per_step', type=int, default=1,
                        help='Maximum number of actions per step')
    parser.add_argument('--max_action_penalty', type=float, default=0,
                        help='Penalty for exceeding the maximum number of actions per step')
    parser.add_argument('--format_reward', type=float, default=0,
                        help='Reward for correct formatting')
    parser.add_argument('--format_penalty', type=float, default=0,
                        help='Penalty for incorrect formatting')
    

    args = parser.parse_args()
    args.name = 'spatial_qa'
    args.env_config = {}
    args.interface_config = {
        'max_action_per_step': args.max_action_per_step,
        'max_action_penalty': args.max_action_penalty,
        'format_reward': args.format_reward,
        'format_penalty': args.format_penalty,
    }
    creator = SpatialQADatasetCreator(config=vars(args))

    creator.create_dataset(
        data_file=args.qa_data_file,
        train_ratio=args.train_ratio,
        force_gen=args.force_gen)
