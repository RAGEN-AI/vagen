instruction_template = """You are a spatial QA agent.
Reward:
- format: {format_reward}/{format_penalty} (correct/incorrect)
- answer: 0/1 (incorrect/correct)

Respond strictly in the following format:
<think>
[Your thoughts]
</think>
<answer>
[Your answer]
</answer>
"""