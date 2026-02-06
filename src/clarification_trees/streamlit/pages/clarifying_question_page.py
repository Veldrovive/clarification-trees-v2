"""
Simple demo for the clarifying question model.

Allows user to create answers to the clarifying questions and then has the model produce new clarifying questions.
"""

import streamlit as st
from PIL import Image

from clarification_trees.streamlit.utils import load_clarification_model, load_answer_model, load_clearvqa_dataset
from clarification_trees.streamlit.components import dialog_trajectory_component
from clarification_trees.dialog_tree import DialogTree, NodeType

@st.dialog("Upload Custom Sample")
def show_custom_input_modal():
    """
    Modal to allow user to upload an image and enter a question.
    Updates the session state dialog tree upon submission.
    """
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    custom_question = st.text_input("Enter ambiguous question")
    
    if st.button("Submit Custom Sample"):
        if uploaded_file is not None and custom_question:
            # Process image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Create new tree
            tree = DialogTree(custom_question, image)
            
            # Update session state
            st.session_state["cqp_dialog_tree"] = tree
            st.session_state["cqp_dialog_tree_leaf"] = DialogTree.ROOT
            
            print(f"Rerunning after custom input. Tree: {st.session_state['cqp_dialog_tree']}")
            st.rerun()
        else:
            st.error("Please provide both an image and a question.")

def clarifying_question_page():
    if st.session_state["current_page"] != "clarifying_question":
        st.session_state["current_page"] = "clarifying_question"

        st.session_state["cqp_sample_index"] = None
        st.session_state["cqp_dialog_tree"] = None
        st.session_state["cqp_dialog_tree_leaf"] = None

        st.session_state["cqp_diverse_predictions"] = None
        
        print("Switched to clarifying question page")

    cfg = st.session_state["cfg"]
    dataset = load_clearvqa_dataset(cfg)

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_index = st.number_input("Sample Index", min_value=0, max_value=len(dataset)-1, key="cqp_sample_index_input")
    
    with col2:
        # Button to trigger the modal
        if st.button("Upload Custom"):
            show_custom_input_modal()

    col1, col2 = st.columns([3, 1])
    with col1:
        produce_diverse_outputs = st.checkbox("Produce Diverse Outputs", value=False)
        if not produce_diverse_outputs:
            st.session_state["cqp_diverse_predictions"] = None
    with col2:
        num_diverse_outputs = st.number_input("Number of Diverse Outputs", min_value=2, max_value=20, value=5, disabled=not produce_diverse_outputs)

    if selected_index != st.session_state["cqp_sample_index"]:
        # If the user touches the number input, this condition becomes True
        # (even if they were previously in custom mode with index -1), 
        # switching them back to dataset mode.
        print(f"New sample selected in clarifying question page: {selected_index}")
        st.session_state["cqp_sample_index"] = selected_index
        
        # We construct a new dialog tree for the selected sample
        sample = dataset[selected_index]
        image = sample[0]
        ambiguous_question = sample[1]["blurred_question"]
        img_caption = sample[1]["caption"]
        unambiguous_question = sample[1]["question"]
        gold_answer = sample[1]["gold_answer"]
        answers = sample[1]["answers"]
        # clarifying_question = sample[1]["clarification_question"]

        tree = DialogTree(ambiguous_question, image, init_image_caption=img_caption, unambiguous_question=unambiguous_question, gold_answer=gold_answer, answers=answers)
        # Add the target node (Assistant's response)
        # cq = tree.add_node(DialogTree.ROOT, NodeType.CLARIFICATION_QUESTION, None, clarifying_question)
        
        st.session_state["cqp_dialog_tree_leaf"] = DialogTree.ROOT
        st.session_state["cqp_dialog_tree"] = tree

    dialog_tree: DialogTree | None = st.session_state["cqp_dialog_tree"]
    dialog_tree_leaf: int | None = st.session_state["cqp_dialog_tree_leaf"]

    if dialog_tree is None:
        st.write("Please select a sample")
        return
    assert dialog_tree_leaf is not None
    
    dialog_trajectory = dialog_tree.get_trajectory(dialog_tree_leaf)
    # is_user_response_turn = dialog_trajectory.trajectory[0].node_type == NodeType.CLARIFICATION_QUESTION
    user_response, generate = dialog_trajectory_component(dialog_trajectory, allow_text_input=True, allow_generate_input=True)

    clarification_model = load_clarification_model(cfg)
    answer_model = load_answer_model(cfg)
    # st.write(model)
    # if dialog_trajectory.trajectory[0].node_type != NodeType.CLARIFICATION_QUESTION:
    if generate:
        if dialog_trajectory.trajectory[0].node_type == NodeType.CLARIFICATION_QUESTION:
            # Then we are generating an answer
            # st.error("Cannot generate an answerer to a clarifying question yet")
            print(f"Last node type: {dialog_trajectory.trajectory[0].node_type}. Generating answer to question {dialog_trajectory.trajectory[0].response}")
            base_system_prompt = cfg.answer_model.base_prompt
            full_system_prompt = f"""
You are pretending to be a human in a dialog with an AI assistant.
Your goal: Answer the assistant's clarifying question to help them realize you are asking about: "{dialog_tree.unambiguous_question}".

Constraints:
1. Answer the clarifying question honestly.
2. Include only minimal information to help the assistant. You are training the assistant to ask good questions.
3. CRITICAL: Do NOT simply state the answer to the unambiguous question (e.g., do not say the number, name, or specific value).
4. Only describe the location, color, or type of object you are interested in.
5. Answer vague questions with vague answers. Provide only the information that was asked for.

Examples:
Image - A bowl of fruit on a table with a red apple on the left, a green apple on the right, a banana, and an orange.
Unambiguous Question - "What color is the apple on the right"
Ambiguous Question - "What color is it?"
Allowed Answers - ["green"]

Clarifying Question - "Are you talking about an apple"
Good Answer - "Yes"
Bad Answer - "Yes, the apple on the right"
Additional information is witheld.

Clarifying Question - "Are you talking about the apple on the left"
Good Answer - "No" or "No, I am talking about the apple on the right"
Question is asking about location, so location can be included in the answer.

Current Context:
Allowed Answers: {dialog_tree.answers}

Given next is the full conversation:
            """.strip()
            # Ambiguous Question: "{dialog_trajectory.trajectory[-1].response}"
            # Assistant's Clarification: "{dialog_trajectory.trajectory[0].response}"
            # Image caption: {dialog_tree.init_image_caption}
            # Gold Answer: {dialog_tree.gold_answer}
            # All Answers: {dialog_tree.answers}
            with st.spinner("Generating answer"):
                prediction = answer_model.generate(dialog_trajectory, base_prompt_override=full_system_prompt, as_user=True)
                if produce_diverse_outputs:
                    diverse_predictions = answer_model.generate_diverse(dialog_trajectory, num_samples=num_diverse_outputs, base_prompt_override=full_system_prompt, as_user=True)
                    st.session_state["cqp_diverse_predictions"] = diverse_predictions

            ca = dialog_tree.add_node(dialog_tree_leaf, NodeType.CLARIFYING_ANSWER, None, prediction[0])
            st.session_state["cqp_dialog_tree_leaf"] = ca
            st.rerun()
        else:
            print(f"Last node type: {dialog_trajectory.trajectory[0].node_type}. Generating clarifying question to question {dialog_trajectory.trajectory[0].response}")
            with st.spinner("Generating clarifying question"):
                prediction = clarification_model.generate(dialog_trajectory)
                if produce_diverse_outputs:
                    diverse_predictions = clarification_model.generate_diverse(dialog_trajectory, num_samples=num_diverse_outputs)
                    st.session_state["cqp_diverse_predictions"] = diverse_predictions
            cq = dialog_tree.add_node(dialog_tree_leaf, NodeType.CLARIFICATION_QUESTION, None, prediction[0])
            st.session_state["cqp_dialog_tree_leaf"] = cq
            st.rerun()

    if user_response is not None:
        print(f"User response: {user_response}")
        if dialog_trajectory.trajectory[0].node_type == NodeType.CLARIFICATION_QUESTION:
            ca = dialog_tree.add_node(dialog_tree_leaf, NodeType.CLARIFYING_ANSWER, None, user_response)
            st.session_state["cqp_dialog_tree_leaf"] = ca
            st.rerun()
        else:
            cq = dialog_tree.add_node(dialog_tree_leaf, NodeType.CLARIFICATION_QUESTION, None, user_response)
            st.session_state["cqp_dialog_tree_leaf"] = cq
            st.rerun()

    if "cqp_diverse_predictions" in st.session_state and st.session_state["cqp_diverse_predictions"] is not None:
        with st.expander("Diverse Predictions"):
            st.write(st.session_state["cqp_diverse_predictions"])
