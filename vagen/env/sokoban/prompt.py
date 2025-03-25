instruction_template = """You are a Sokoban solver. Push all boxes (X) onto targets (O).

Symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target | S You on Target

Rules:
- Push boxes only (no pulling)
- Use actions: Up, Down, Left, Right (max {max_action_per_step} per turn)

Rewards: -0.1 per move | +1.0 box on target | +10.0 all boxes placed | Format: {format_reward}/{format_penalty} (correct/incorrect)

Respond with:
<think>...</think><answer>...</answer>
"""


init_observation_template = """
[Initial Observation]:
{observation}
Decide your next action(s).
Respond with:
<think>...</think><answer>...</answer>
"""

valid_action_template = """Valid action extracted from your response is {valid_action}.\
After that, the observation is:
{observation}
reward: {reward}
done: {done}
Decide your next action(s).
Respond with:
<think>...</think><answer>...</answer>
"""

invalid_action_template = """Invalid response. You stay at the same position.
The observation is:
{observation}
reward: {reward}
done: {done}
Decide your next action(s).
Respond with:
<think>...</think><answer>...</answer>
"""