system_prompt = """
You are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer.
"""

instruction_template = """You are a Sokoban solver.

Sokoban Quick Guide
Goal: Push all boxes (X) onto targets (O).

Symbols:
# Wall | _ Floor | O Target | X Box | P You | √ = Box on Target | S = You on Target
The observation is a 2D grid of the current state of the Sokoban game.

Rules:
1. Push boxes (can't pull).
2. Avoid walls (#).

Actions you can take: Up, Down, Left, Right. You can take up to {max_action_per_step} action(s) at a time.
- Up: move up to the cell above
- Down: move down to the cell below
- Left: move left to the cell to the left
- Right: move right to the cell to the right
If there is a box on the cell you want to move to, you will push the box one cell in the same direction.

Rewards:
One move: -0.1
Box on target: +1.0
All boxes placed: +10.0
Format incorrect: {format_penalty}
Format correct: {format_reward}

Please think step by step and provide the action(s) you want to take.
Wrap your thought between `<think>` and `</think>` tags, then wrap the action(s) between `<answer>` and `</answer>` tags.
STRICTLY follow the format:
<think>...</think><answer>...</answer>
"""
# E.g. <think> There's a box on the upper right of me, the target is on the upper side of the box, I need to push the box it upward. </think><answer> Right,Up,Up </answer>
# Let's try to use a format reward and answer reward
# If the reponse provides a final answer and is corect, the model receives an accurtacy reward of +1
# is the response encloses its thinking in <think></think> and the final answer is <answer></answer> tags, the model receives a format reward of +1



init_observation_template = """
[Initial Observation]:
{observation}
Decide your next action(s).
"""

valid_action_template = """Valid action extracted from your answer is {valid_action}.\
After that, the observation is:
{observation}
Decide your next action(s).
"""

invalid_action_template = """Invalid response. You stay at the same position.
The observation is:
{observation}
Decide your next action(s).
STRICTLY follow the format:
<think>...</think><answer>...</answer>
"""