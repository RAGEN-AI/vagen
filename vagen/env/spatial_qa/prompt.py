instruction_template = """You are a spatial QA agent. You need to answer the question based on the given information.
Reward:
- format: {format_reward}/{format_penalty} (correct/incorrect)
- answer: 0/1 (incorrect/correct)

Please strictly follow the given response format, respond with:
<think>...</think><answer>...</answer>
"""