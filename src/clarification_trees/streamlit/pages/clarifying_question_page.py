"""
Simple demo for the clarifying question model.

Allows user to create answers to the clarifying questions and then has the model produce new clarifying questions.
"""

import streamlit as st
from PIL import Image

from clarification_trees.streamlit.utils import load_clarification_model, load_clearvqa_dataset
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
        # clarifying_question = sample[1]["clarification_question"]

        tree = DialogTree(ambiguous_question, image)
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

    model = load_clarification_model(cfg)
    # st.write(model)
    # if dialog_trajectory.trajectory[0].node_type != NodeType.CLARIFICATION_QUESTION:
    if generate:
        print(f"Last node type: {dialog_trajectory.trajectory[0].node_type}. Generating clarifying question to question {dialog_trajectory.trajectory[0].response}")
        with st.spinner("Generating clarifying question"):
            prediction = model.generate(dialog_trajectory)
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
