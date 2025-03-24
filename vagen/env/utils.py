import re
from PIL import Image
import numpy as np
from dataclasses import dataclass
from vagen.env.base import BaseEnv, IMAGE_PLACEHOLDER
from typing import List, Dict, Tuple, Union



def preprocess_text(text: str, strict_match: bool = True) -> dict:
    """Preprocess the raw text from llm to a list of strings.
    
    Args:
        text: raw text from llm
        strict_match: If True, enforces strict format where text must contain exactly one 
                     <think>...</think> followed by one <answer>...</answer> with possible 
                     whitespace between them. No content should appear before <think> or 
                     after </answer>.

    Returns:
        dict with keys: llm_raw_response, think, answer_list
    """
    
    if strict_match:
        strict_match_result = None

        # Verify exactly one occurrence of each tag
        think_open_count = text.count("<think>")
        think_close_count = text.count("</think>")
        answer_open_count = text.count("<answer>")
        answer_close_count = text.count("</answer>")
        tags_are_balanced = (think_open_count == 1 and think_close_count == 1 and 
                            answer_open_count == 1 and answer_close_count == 1)
        
        if tags_are_balanced:
            strict_format_pattern = r'^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$'
            strict_match_result = re.match(strict_format_pattern, text, re.DOTALL)
        if strict_match_result is None:
            return {
                'llm_raw_response': text,
                'answer_list': [],
                'think': "",
                'answer': "",
            }
    
    # Extract content from <think> tags
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    
    # Extract content from <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)

    answer_list, thinking, answer_content = [], "", ""
    
    if think_match:
        thinking = think_match.group(1).strip()
    
    if answer_match:
        # Get the answer content and split by comma
        answer_content = answer_match.group(1).strip()
        # Split by comma and strip whitespace from each item
        answer_list = [item.strip() for item in answer_content.split(',') if item.strip()]
    
    return {
        'llm_raw_response': text,
        'answer_list': answer_list,
        'think': thinking,
        'answer': answer_content,
    }



def convert_numpy_to_PIL(numpy_array: np.ndarray) -> Image.Image:
        """Convert a numpy array to a PIL RGB image."""
        if numpy_array.shape[-1] == 3:
            # Convert numpy array to RGB PIL Image
            return Image.fromarray(numpy_array, mode='RGB')
        else:
            raise ValueError(f"Unsupported number of channels: {numpy_array.shape[-1]}. Expected 3 (RGB).")



@dataclass
class PreprocessResult:
    action_list: List # list of valid action defined in the action space
    answer_list: List[str] # string of extracted answer (may be invalid action)
    think: str
    answer: str
    llm_raw_response: str

    def to_dict(self):
        return {
            'action_list': self.action_list,
            'answer_list': self.answer_list,
            'think': self.think,
            'answer': self.answer,
            'llm_raw_response': self.llm_raw_response,
        }

def preprocess(text: str, extract_action_func, invalid_action_code=0) -> PreprocessResult:
    """Preprocess the raw text from LLM into a list of actions.
    
    Args:
        text: Raw text from LLM
        extract_action_func: Function to extract action from text, should return action ID or invalid_action_code
            Function signature: extract_action_func(text: str) -> int
        invalid_action_code: Code representing an invalid action (default: 0)

    Returns:
        PreprocessResult containing parsed valid actions
    """
    parsed_text = preprocess_text(text)
    
    # Process actions until first invalid action
    action_list = []
    for answer in parsed_text['answer_list']:
        action = extract_action_func(answer)
        if action == invalid_action_code:
            break
        action_list.append(action)
    
    return PreprocessResult(
        action_list=action_list,
        answer_list=parsed_text['answer_list'],
        think=parsed_text['think'],
        answer=parsed_text['answer'],
        llm_raw_response=text
    )


def postprocess(
    env_state: Union[str, np.ndarray], 
    reward: float,
    done: bool,
    info: Dict,
    preprocess_result: PreprocessResult,
    valid_action_template: str,
    invalid_action_template: str,
    action_lookup: Dict = None,
) -> Tuple[Dict, float, bool, Dict]:
    """Postprocess the environment feedback to obs, reward, done, info
    NOTE now assume there's only one image in the observation

    Args:
        env_state: environment state (text or numpy array (image))
        reward: reward of the environment
        done: whether the environment is done
        info: extra info
        preprocess_result: preprocess result
        action_lookup: action lookup to convert action space to text
        valid_action_template: text template for valid action
        invalid_action_template: text template for invalid action

    Returns:
        Tuple[Dict, float, bool, Dict]: observation, reward, done, info
    """

    if isinstance(env_state, np.ndarray):
        env_state = convert_numpy_to_PIL(env_state)

    answer = preprocess_result.answer
    valid_action = []
    for action in preprocess_result.action_list:
        if action_lookup is None:
            valid_action.append(action)
        else:
            valid_action.append(action_lookup[action])

    observation = IMAGE_PLACEHOLDER if not isinstance(env_state, str) else env_state
    if len(valid_action) > 0:
        text_template = valid_action_template.format(
            valid_action=valid_action,
            observation=observation,
            reward=reward,
            done=done,
        )
    else:
        text_template = invalid_action_template.format(
            observation=observation,
            reward=reward,
            done=done,
        )

    if isinstance(env_state, str):
        obs = {'text_template': text_template}
    else:
        obs = {
            'text_template': text_template,
            'multi_modal_data': {
                IMAGE_PLACEHOLDER: [env_state],
            },
        }
    return obs, reward, done, info








if __name__ == "__main__":
    text = """
    <think>
    I am thinking about the problem.
    </think>
    <answer>
    answer1, answer2, answer3
    </answer>
    """
    print(preprocess_text(text))