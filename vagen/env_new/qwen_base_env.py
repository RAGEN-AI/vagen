from abc import ABC, abstractmethod
import re
from typing import Optional, List, Tuple, Any, Dict
from copy import deepcopy
from transformers import AutoTokenizer
import torch
from PIL import Image
from dataclasses import dataclass, field

class BaseEnv(ABC):
    @abstractmethod
    def _reset(self, seed: Optional[int] = None) -> Any:
        """
        Reset the environment.
        NOTE: the environment should be same for the same seed
        Args:
            seed: Seed for the environment
            
        Returns:
            rendered environment
        """
        pass
    
    @abstractmethod
    def _step(self, action) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
            
        Returns:
            observation (rendered environment), reward, done, info
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close the environment."""
        pass
    
    
    def step(self, action:Any) -> Tuple[Any, Any, Any, Any]:
        """
        Execute one step in the environment.
        Args:
            action: Action to take, must be in action space, or default invalid action
            
        Returns:
            observation (rendered environment), reward, done, info
        """
        obs,reward,done,info = self._step(action)
        return obs, reward, done, info
    
    def reset(self, seed: Optional[int] = None) -> Any:
        """
        Reset the environment.
        NOTE: the environment should be same for the same seed
        Args:
            seed: Seed for the environment  
        Returns:
            obs,info
        """
        obs,info = self._reset(seed)
        return obs,info
    
        
class BaseInterface(ABC):
    def __init__(self, **env_config):
        self.env_config = env_config
        
    @classmethod
    def name_repr(cls) -> str:
        """Get the name of the environment."""
        return cls.__name__
        
    @abstractmethod
    def _reset(self, seed: Optional[int] = None) -> Tuple[Any, float, bool, Dict]:
        """Reset the environment."""
        pass
    
    @abstractmethod
    def _step(self, action:str) -> Tuple[Any, float, bool, Dict]:
        """Execute action string in the environment."""
        # return observation, reward, done, info
        # info must contain "llm_raw_response" key, which is a string
        pass
    
    @classmethod
    @abstractmethod
    def config_repr(cls, config: Dict) -> str:
        """Get the config of the environment."""
        pass
    
    
    @abstractmethod
    def close(self):
        """Close the environment."""
        pass
    
    @abstractmethod
    def get_task_instruction(self) -> str:
        """Get the task instruction."""
        pass
    
    
    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        """Execute action string in the environment."""
        """Please use the following assertions to validate the output, 
        then you can rewrite the step in your own class to improve the performance"""
        
        
        assert isinstance(action, str), f"action must be str, got {type(action)}"
        obs,reward,done,info = self._step(action)
        assert isinstance(reward, (int, float)), f"reward must be int or float, got {type(reward)}"
        assert isinstance(done, bool), f"done must be bool, got {type(done)}"
        assert isinstance(info, dict), f"info must be dict, got {type(info)}"
        assert isinstance(obs, dict), f"obs must be dict, got {type(obs)}"
        assert "llm_raw_response" in info, f"info must contain 'llm_raw_response' key"
        assert isinstance(info["llm_raw_response"], str), f"info['llm_raw_response'] must be str, got {type(info['llm_raw_response'])}"
        assert "text_template" in obs, f"obs must contain 'text_template' key"
        assert isinstance(obs["text_template"], str), f"obs['text_template'] must be str, got {type(obs['text_template'])}"
        
        
        
        if "obs_image" in obs:
            for image in obs["obs_image"]:
                assert isinstance(image, Image.Image), f"image must be PIL.Image.Image, got {type(image)}"
        return obs, reward, done, info
    
            
    