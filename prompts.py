evaluation_prompt = """
Please evaluate the following response based on {criteria}. 

Provide a detailed analysis including:
1. Score (1-10)
2. Detailed explanation of the score
3. Specific strengths
4. Areas for improvement
5. Key observations

Response to evaluate:
{response}

Format your response as JSON:
{{
    "score": <score>,
    "explanation": "<detailed explanation>",
    "strengths": ["<strength1>", "<strength2>", ...],
    "improvements": ["<improvement1>", "<improvement2>", ...],
    "observations": "<key observations>"
}}
"""


comparison_prompt = """
Compare these two responses and provide a detailed analysis.

Response 1:
{response1}

Response 2:
{response2}

Please provide:
1. Which response is better (1 or 2)
2. A detailed explanation of why it's better
3. Point-by-point comparison of key aspects
4. Specific examples from both responses
5. Suggestions for improving both responses

Format your response as JSON:
{{
    "winner": "1 or 2",
    "explanation": "<detailed explanation>",
    "comparison_points": [
        {{"aspect": "<aspect>", "response1": "<analysis1>", "response2": "<analysis2>"}},
        ...
    ],
    "improvement_suggestions": {{
        "response1": ["<suggestion1>", ...],
        "response2": ["<suggestion1>", ...]
    }}
}}
"""