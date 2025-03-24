import numpy as np
import re
import copy
from typing import Tuple, Dict, Optional, List, Any, Union
from PIL import Image
import json

from vagen.env.register import register
from vagen.env.base import (
    BaseEnv,
    BaseInterface
)

from vagen.env.utils import preprocess
from vagen.env.spatial_qa.QA import QA
from vagen.env.spatial_qa.prompt import instruction_template

"""
For each environment, we use env_config to create the env. One env_config corresponds to a set of envs.
- For Sokoban, env_config set the size, n_box, ... of the environment.
- For spatial_qa, env_config contains data path, also corresponds to a set of envs (QAs).

We use seed to pick one env from the set
- For Sokoban, seed is used to as a random seed to generate the room
- For spatial_qa, seed is used as an index to pick one QA from the list
"""


class SpatialQAEnv(BaseEnv):
    """
    SpatialQAEnv is a QA gym-like environment
    Each environment will 
    """

    def __init__(self, data_path: str, type: str = None):
        BaseEnv.__init__(self)
        self.data_path = data_path
        with open(data_path, 'r') as f:
            self.QAs = json.load(f)['QAs']
        self.qa = None
        self.is_finished = False
        self.is_correct = False
        self.type = type

    def _reset(self, seed: int):
        # NOTE seed is expected to be in [0, len(self.QAs)), but in case of unexpected input, we take modulo
        if seed < 0 or seed >= len(self.QAs):
            print(f"[WARNING] seed {seed} is out of range [0, {len(self.QAs)}), taking modulo")
            seed = seed % len(self.QAs)

        qa_dict = self.QAs[seed]
        qa = QA.from_dict(qa_dict)
        if self.type is not None:
            assert qa.type == self.type, f"[ERROR] type mismatch, expected {self.type}, got {qa.type}"
        
        self.qa = qa
        self.is_finished = False
        self.is_correct = False
        return qa.question, {}
    
    def _step(self, action: str):
        assert not self.is_finished
        if_correct = self.qa.evaluate(action)
        self.is_correct = if_correct
        self.is_finished = True
        return "You have finished the task", if_correct, self.is_finished, {}
     

    def _render(self, mode='text'):
        assert mode == 'text'
        if not self.is_finished:
            return self.qa.question
        else:
            return ""

    def close(self):
        pass

    def finished(self):
        return self.is_finished
    
    def success(self):
        return self.is_finished and self.is_correct








@register(name="spatial_qa")
class SpatialQAInterface(BaseInterface):

    INVALID_ACTION = ""

    def __init__(
            self,
            env_config: Dict,
            interface_config: Dict,
        ):
        """
        Args:
            env_config (Dict): used to create the env
                - type: type of the spatial QA
                - data_path: path to the data file
            interface_config (Dict): interface configuration
        """
        qa_type = env_config['type']
        data_path = env_config['data_path']
        self.env = SpatialQAEnv(data_path=data_path, type=qa_type)

        max_action_per_step = interface_config.setdefault('max_action_per_step', 1)
        max_action_penalty = interface_config.setdefault('max_action_penalty', 0.0)
        format_reward = interface_config.setdefault('format_reward', 0.0)
        format_penalty = interface_config.setdefault('format_penalty', 0.0)

        self.env_config = {'type': qa_type, 'data_path': data_path}
        self.interface_config = {
            'max_action_per_step': max_action_per_step,
            'max_action_penalty': max_action_penalty,
            'format_reward': format_reward,
            'format_penalty': format_penalty,
        }
        
    @classmethod
    def _extract_one_action(cls, text):
        """Extract single action from text, the input text should ensure only one action contained"""

        return text
    

    def _step(self, raw_text: str) -> Tuple[Any, float, bool, Dict]:
        """Step the environment with llm raw response
        - Multiple actions are allowed, execute until the first invalid action or environment terminates
        - The observation is the last step observation
        
        Args:
            raw_text: raw text from LLM

        Returns:
            Tuple[Any, float, bool, Dict]: observation, reward, done, info
            - observation (dict): observation of the environment
            - reward (float): reward of the environment for the raw_text (multiple actions, including format reward and env reward)
            - done (bool): whether the environment is done
            - info (dict): extra info
        """

        assert not self.env.finished(), "Environment finished before step"
        reward, done, final_info = 0, False, {}


        # preprocess_result = self._preprocess(raw_text)
        preprocess_result = preprocess(raw_text, self._extract_one_action, self.INVALID_ACTION, answer_sep="|") # NOTE we do not accept multiple actions here, so we use | as the answer separator
        think = preprocess_result.think
        action_list = preprocess_result.action_list
        answer = preprocess_result.answer
        final_info['llm_raw_response'] = preprocess_result.llm_raw_response


        # parse format and action list
        if not action_list:
            reward += self.interface_config['format_penalty']
            env_state = "Invalid answer"
            done = True
            info = {}
            
        else:
            reward += self.interface_config['format_reward']
            if len(action_list) > self.interface_config['max_action_per_step']:
                reward += self.interface_config['max_action_penalty']
                action_list = action_list[:self.interface_config['max_action_per_step']]
                preprocess_result.action_list = action_list
            _, env_reward, done, info = self.env.step(action_list[0])
            reward += env_reward
            env_state = self.env._render(mode='text')

        self.traj_reward += reward
        final_info.update(info) # NOTE currently only use the last step info
        return {"text_template": env_state}, reward, done, final_info
    
    def _reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment and return the observation at the first step.
        """
        self.env._reset(seed=seed)
        self.traj_reward = 0
        env_state = self.env._render(mode='text') # question
        return {"text_template": env_state}, {}

    def close(self):
        self.env.close()

    @classmethod
    def config_repr(cls, env_config: Dict, interface_config: Dict) -> str:
        """
        Create a string representation of the configuration.
        
        Args:
            env_config: Dictionary containing environment configuration
            interface_config: Dictionary containing interface configuration
            
        Returns:
            String representation of the configuration
            
        Raises:
            ValueError: If required keys are missing from the configuration
        """

        env_config_str = f"SpatialQA(type={env_config['type']}, data_path={env_config['data_path']})"
        interface_config_str = (
            f"SpatialQAInterface(max_action_per_step={interface_config.get('max_action_per_step', 1)}, "
            f"max_action_penalty={interface_config.get('max_action_penalty', 0.0)}, "
            f"format_reward={interface_config.get('format_reward', 0.0)}, "
            f"format_penalty={interface_config.get('format_penalty', 0.0)})"
        )
        return f"{env_config_str}, {interface_config_str}"
    def get_task_instruction(self) -> str:
        return instruction_template.format(
            format_reward=self.interface_config['format_reward'],
            format_penalty=self.interface_config['format_penalty'],
        )
    
    def get_traj_reward(self):
        return self.traj_reward

    def success(self):
        return self.env.success()
