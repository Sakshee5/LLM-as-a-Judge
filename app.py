import streamlit as st
from openai_utils import get_response
from prompts import criteria_based_evaluation_prompt, reference_based_eval_prompt, comparison_prompt, detect_hallucinations

def initialize_session_states():
    if 'judge_model' not in st.session_state:
        st.session_state.judge_model = "gpt-4o"

    if 'pc_response1' not in st.session_state:
        st.session_state.pc_response1 = ""

    if 'pc_response2' not in st.session_state:
        st.session_state.pc_response2 = ""

    if 'crf_response' not in st.session_state:
        st.session_state.crf_response = ""

    if 'rbe_response' not in st.session_state:
        st.session_state.rbe_response = ""
    
    if 'dh_response' not in st.session_state:
        st.session_state.dh_response = ""



# Model selection
available_models = ["gpt-4", "gpt-3.5-turbo"]

def pairwise_comparison():
    """Pairwise comparison interface"""
    st.subheader("Pairwise Comparison")
    # Input prompt
    prompt = st.text_area("Enter your prompt:")
    
    # Model selection for comparison
    st.subheader("Select Models to Compare")
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("Model 1", available_models, key="model1")
    with col2:
        model2 = st.selectbox("Model 2", available_models, key="model2")

    if prompt and st.button("Generate"):
        # Get responses
        with st.spinner("Generating response for Model 1..."):
            st.session_state.pc_response1 = get_response(prompt, model1, json_format=False)
        with st.spinner("Generating response for Model 2..."):
            st.session_state.pc_response2 = get_response(prompt, model2, json_format=False)

    if st.session_state.pc_response1 and st.session_state.pc_response2 and prompt:   
        # Display responses
        st.subheader("Model Responses")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{model1} Response:**")
            st.write(st.session_state.pc_response1)
        with col2:
            st.write(f"**{model2} Response:**")
            st.write(st.session_state.pc_response2)
    
    if st.session_state.pc_response1 and st.button('Evaluate'):
        with st.spinner("Evaluating Responses..."):
            comparison_result = get_response(comparison_prompt.format(response1=st.session_state.pc_response1, response2=st.session_state.pc_response2), st.session_state.judge_model)
        
        st.subheader("Winner")
        st.write(f"Model {comparison_result['winner']}")
        st.subheader(f"**Detailed Explanation:**")
        st.write(comparison_result['explanation'])
        
        st.subheader("\n**Point-by-Point Comparison:**")
        for point in comparison_result['comparison_points']:
            with st.expander(f"Comparison: {point['aspect']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{model1}:**")
                    st.write(point['response1'])
                with col2:
                    st.write(f"**{model2}:**")
                    st.write(point['response2'])
        
        st.subheader("\n**Improvement Suggestions:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{model1} Improvements:**")
            for suggestion in comparison_result['improvement_suggestions']['response1']:
                st.write(f"- {suggestion}")
        with col2:
            st.write(f"**{model2} Improvements:**")
            for suggestion in comparison_result['improvement_suggestions']['response2']:
                st.write(f"- {suggestion}")


def evaluation_by_criteria_ref_free():
    
    st.subheader("Criteria based Reference Free Evaluation")
    # Input prompt
    prompt = st.text_area("Enter your prompt:")
    criteria = st.text_input("Enter comma seperated criteria to evaluate on", value ="accuracy, coherence, creativity, relevance")
    
    criteria_list = criteria.split(',')
    # Model selection for comparison
    st.subheader("Select Model to Evaluate")  
    model = st.selectbox("Select Model", available_models, key="model")

    if prompt and st.button("Generate"):
        # Get responses
        with st.spinner("Generating response ..."):
            st.session_state.crf_response = get_response(prompt, model, json_format=False)

    if st.session_state.crf_response:
        st.write(f"**{model} Response:**")
        st.write(st.session_state.crf_response)
    
    if st.session_state.crf_response and st.button("Evaluate"):
        # Detailed evaluation section
        st.subheader("Detailed Evaluation for each criteria")
        for cri in criteria_list:
            with st.spinner(f"Evaluating Responses for {cri}..."):
                eval_result = get_response(criteria_based_evaluation_prompt.format(criteria=cri, response=st.session_state.crf_response), st.session_state.judge_model)

            if eval_result:
                # Display detailed evaluation
                with st.expander(f"{cri.capitalize()} Analysis"):
                    st.write(f"**Score:** {eval_result['score']}/10")
                    st.write(f"**Detailed Explanation:** {eval_result['explanation']}")
                    
                    st.write("\n**Strengths:**")
                    for strength in eval_result['strengths']:
                        st.write(f"- {strength}")
                        
                    st.write("\n**Areas for Improvement:**")
                    for improvement in eval_result['improvements']:
                        st.write(f"- {improvement}")
                        
                    st.write(f"\n**Key Observations:** {eval_result['observations']}")


def reference_based_evaluation():
    """Evaluate responses against a reference/ground truth answer"""
    st.subheader("Reference-Based Evaluation")
    
    # Input fields
    prompt = st.text_area("Enter your prompt:", value="What is the primary purpose of backpropagation algorithm in neural networks?")
    reference_answer = st.text_area("Enter reference answer:", value="To calculate gradients and update weights to minimize errors")
    model = st.selectbox("Select Model", available_models, key="ref_model")
    
    if prompt and reference_answer and st.button("Generate"):
        with st.spinner("Generating response..."):
            st.session_state.rbe_response = get_response(prompt, model, json_format=False)
            
    if st.session_state.rbe_response:
        st.write(f"**{model} Response:**")
        st.write(st.session_state.rbe_response)
        
    if st.session_state.rbe_response and st.button("Evaluate"):
        with st.spinner("Generating evaluation..."):
            eval_result = get_response(reference_based_eval_prompt.format(reference_answer=reference_answer, model_response=st.session_state.rbe_response), st.session_state.judge_model)
        
        if eval_result:
            st.write(f"**Score:** {eval_result['score']}/10")
            st.write(f"**Detailed Explanation:** {eval_result['explanation']}")


def detect_hallucination_eval():
    with open("QA_context.txt", mode='r', encoding='utf-8') as f:
        context = f.read()

    question = st.text_input("Ask question about Apples Financial Statements for 2024", value="What is the percentage change in Apple's term debt (current + non-current) from September 30, 2023, to September 28, 2024? While calculating, make a minor error that a huamn might not be able to see immediately.")
    model = st.selectbox("Select Model", available_models, key="ref_model")

    if question and st.button("Generate"):
        with st.spinner("Generating response..."):
            st.session_state.dh_response = get_response(f"Context: {context}\n\n Question: {question}\n\nAlso mention the data/source that helped you asnwer the question.", model, json_format=False)

    if st.session_state.dh_response:
        st.write(f"**{model} Response:**")
        st.write(st.session_state.dh_response)
        
    if st.session_state.dh_response and st.button("Evaluate"):
        with st.spinner("Generating evaluation..."):
            eval_result = get_response(detect_hallucinations.format(context = context, question=question, response=st.session_state.dh_response), st.session_state.judge_model)
        
        if eval_result:
            st.write(f"**Detailed Explanation:** {eval_result['explanation']}")
            st.write(f"**Correct Answer:** {eval_result['correct answer']}")


def main():
    initialize_session_states()

    st.title("LLM-As-a-Judge")

    evaluation_methods = {
        "Pairwise Comparison": pairwise_comparison,
        "Reference-Free Criteria Evaluation": evaluation_by_criteria_ref_free,
        "Reference-based Evaluation": reference_based_evaluation,
        "Hallucination Detection": detect_hallucination_eval,
    }
    
    method = st.sidebar.selectbox(
        "Select Evaluation Method",
        list(evaluation_methods.keys())
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Method Description")
    
    descriptions = {
        "Pairwise Comparison": "Compare two LLM responses directly to determine which is better",
        "Reference-Free Criteria Evaluation": "Evaluate as per a defined criteria without ground truth",
        "Reference-based Evaluation": "Evaluate responses against a reference/ground truth answer",
        "Hallucination Detection": "Evaluate response for hallucinations.",
    }
    
    st.sidebar.write(descriptions[method])
    
    evaluation_methods[method]()

if __name__ == "__main__":
    main()