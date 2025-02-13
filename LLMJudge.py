import streamlit as st
import openai
from typing import List, Dict
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import streamlit as st
import openai
from typing import List, Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

def setup_openai_client(api_key):
    openai.api_key = api_key
    return openai

def get_llm_response(prompt: str, model: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def evaluate_response(response: str, criteria: str, judge_model: str) -> Dict:
    """Evaluate a response with detailed reasoning"""
    evaluation_prompt = f"""
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
    
    try:
        eval_response = openai.ChatCompletion.create(
            model=judge_model,
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.3
        )
        return json.loads(eval_response.choices[0].message.content)
    except:
        return {
            "score": 0,
            "explanation": "Evaluation failed",
            "strengths": [],
            "improvements": [],
            "observations": "Evaluation failed"
        }

def pairwise_comparison(response1: str, response2: str, judge_model: str) -> Dict:
    """Perform detailed pairwise comparison between two responses"""
    comparison_prompt = f"""
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
    
    try:
        comparison = openai.ChatCompletion.create(
            model=judge_model,
            messages=[{"role": "user", "content": comparison_prompt}],
            temperature=0.3
        )
        return json.loads(comparison.choices[0].message.content)
    except:
        return {
            "winner": "Error",
            "explanation": "Comparison failed",
            "comparison_points": [],
            "improvement_suggestions": {"response1": [], "response2": []}
        }

# Streamlit UI
st.title("LLM Judge Evaluation Platform")
st.markdown("""
This platform provides detailed analysis and comparison of LLM responses with comprehensive explanations of the evaluation process.
""")

# API Key input
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

if api_key:
    client = setup_openai_client(api_key)
    
    # Model selection
    available_models = ["gpt-3.5-turbo", "gpt-4"]
    judge_model = st.sidebar.selectbox("Select Judge Model", available_models)
    
    # Input prompt
    prompt = st.text_area("Enter your prompt:")
    
    # Model selection for comparison
    st.subheader("Select Models to Compare")
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("Model 1", available_models, key="model1")
    with col2:
        model2 = st.selectbox("Model 2", available_models, key="model2")
    
    if st.button("Generate and Evaluate"):
        if prompt:
            with st.spinner("Generating responses and evaluating..."):
                # Get responses
                response1 = get_llm_response(prompt, model1)
                response2 = get_llm_response(prompt, model2)
                
                # Display responses
                st.subheader("Model Responses")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{model1} Response:**")
                    st.write(response1)
                with col2:
                    st.write(f"**{model2} Response:**")
                    st.write(response2)
                
                # Evaluate responses
                criteria_list = ["accuracy", "coherence", "creativity", "relevance"]
                evaluation_results = []
                
                # Detailed evaluation section
                st.subheader("Detailed Individual Evaluations")
                for model, response in [(model1, response1), (model2, response2)]:
                    st.write(f"\n### {model} Evaluation")
                    model_results = {"model": model}
                    
                    for criteria in criteria_list:
                        eval_result = evaluate_response(response, criteria, judge_model)
                        model_results[criteria] = eval_result["score"]
                        
                        # Display detailed evaluation
                        with st.expander(f"{criteria.capitalize()} Analysis"):
                            st.write(f"**Score:** {eval_result['score']}/10")
                            st.write(f"**Detailed Explanation:** {eval_result['explanation']}")
                            
                            st.write("\n**Strengths:**")
                            for strength in eval_result['strengths']:
                                st.write(f"- {strength}")
                                
                            st.write("\n**Areas for Improvement:**")
                            for improvement in eval_result['improvements']:
                                st.write(f"- {improvement}")
                                
                            st.write(f"\n**Key Observations:** {eval_result['observations']}")
                    
                    evaluation_results.append(model_results)
                
# Create radar chart
                df = pd.DataFrame(evaluation_results)
                
                # Create figure
                fig = go.Figure()
                
                # Add traces for each model
                for model in df['model']:
                    model_data = df[df['model'] == model]
                    fig.add_trace(go.Scatterpolar(
                        r=[model_data[criteria].iloc[0] for criteria in criteria_list],
                        theta=criteria_list,
                        name=model,
                        fill='toself'
                    ))
                
                # Update layout
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )
                    ),
                    showlegend=True,
                    title="Model Evaluation Radar Chart"
                )
                
                st.plotly_chart(fig)

                
                # Detailed pairwise comparison
                st.subheader("Detailed Pairwise Comparison")
                comparison_result = pairwise_comparison(response1, response2, judge_model)
                
                st.write(f"**Winner:** Model {comparison_result['winner']}")
                st.write(f"**Detailed Explanation:**")
                st.write(comparison_result['explanation'])
                
                st.write("\n**Point-by-Point Comparison:**")
                for point in comparison_result['comparison_points']:
                    with st.expander(f"Comparison: {point['aspect']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**{model1}:**")
                            st.write(point['response1'])
                        with col2:
                            st.write(f"**{model2}:**")
                            st.write(point['response2'])
                
                st.write("\n**Improvement Suggestions:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{model1} Improvements:**")
                    for suggestion in comparison_result['improvement_suggestions']['response1']:
                        st.write(f"- {suggestion}")
                with col2:
                    st.write(f"**{model2} Improvements:**")
                    for suggestion in comparison_result['improvement_suggestions']['response2']:
                        st.write(f"- {suggestion}")
                
        else:
            st.warning("Please enter a prompt to evaluate.")
else:
    st.warning("Please enter your OpenAI API key to proceed.")