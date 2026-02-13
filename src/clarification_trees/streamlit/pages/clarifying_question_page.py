"""
Simple demo for the clarifying question model.

Allows user to create answers to the clarifying questions and then has the model produce new clarifying questions.
"""

import streamlit as st
from PIL import Image
from pathlib import Path
from clarification_trees.streamlit.utils import load_clarification_model, load_answer_model, load_semantic_clusterer, load_clearvqa_dataset
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
            tree = DialogTree(custom_question, image, Path(uploaded_file.name))
            
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
        st.session_state["cqp_diverse_semantic_centers"] = None
        
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
            st.session_state["cqp_diverse_semantic_centers"] = None
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
        image = sample.image
        assert image is not None, "ClearVQADataset was created without image loading enabled."
        ambiguous_question = sample.blurred_question
        img_caption = sample.caption
        unambiguous_question = sample.question
        gold_answer = sample.gold_answer
        answers = sample.answers
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
    semantic_clusterer = load_semantic_clusterer(cfg)
    # st.write(model)
    # if dialog_trajectory.trajectory[0].node_type != NodeType.CLARIFICATION_QUESTION:
    if generate:
        if dialog_trajectory.trajectory[0].node_type == NodeType.CLARIFICATION_QUESTION:
            # Then we are generating an answer
            # st.error("Cannot generate an answerer to a clarifying question yet")
            print(f"Last node type: {dialog_trajectory.trajectory[0].node_type}. Generating answer to question {dialog_trajectory.trajectory[0].response}")
            base_system_prompt = cfg.answer_model.answer_base_prompt.strip().format(unambiguous_question=dialog_tree.unambiguous_question, answers=dialog_tree.answers)
            with st.spinner("Generating answer"):
                prediction = answer_model.generate(dialog_trajectory, base_prompt_override=base_system_prompt, as_user=True)
                print(f"Generated answer: {prediction}")
                if produce_diverse_outputs:
                    diverse_predictions = answer_model.generate_diverse(dialog_trajectory, num_samples=num_diverse_outputs, base_prompt_override=base_system_prompt, as_user=True)
                    clustered_diverse_predictions, diverse_semantic_centers = semantic_clusterer.cluster(diverse_predictions)
                    st.session_state["cqp_diverse_predictions"] = clustered_diverse_predictions
                    st.session_state["cqp_diverse_semantic_centers"] = diverse_semantic_centers

            ca = dialog_tree.add_node(dialog_tree_leaf, NodeType.CLARIFYING_ANSWER, prediction[0])
            st.session_state["cqp_dialog_tree_leaf"] = ca
            st.rerun()
        else:
            print(f"Last node type: {dialog_trajectory.trajectory[0].node_type}. Generating clarifying question to question {dialog_trajectory.trajectory[0].response}")
            with st.spinner("Generating clarifying question"):
                prediction = clarification_model.generate(dialog_trajectory)
                if produce_diverse_outputs:
                    diverse_predictions = clarification_model.generate_diverse(dialog_trajectory, num_samples=num_diverse_outputs)
                    clustered_diverse_predictions, diverse_semantic_centers = semantic_clusterer.cluster(diverse_predictions)
                    st.session_state["cqp_diverse_predictions"] = clustered_diverse_predictions
                    st.session_state["cqp_diverse_semantic_centers"] = diverse_semantic_centers
            cq = dialog_tree.add_node(dialog_tree_leaf, NodeType.CLARIFICATION_QUESTION, prediction[0])
            st.session_state["cqp_dialog_tree_leaf"] = cq
            st.rerun()

    if user_response is not None:
        print(f"User response: {user_response}")
        if dialog_trajectory.trajectory[0].node_type == NodeType.CLARIFICATION_QUESTION:
            ca = dialog_tree.add_node(dialog_tree_leaf, NodeType.CLARIFYING_ANSWER, user_response)
            st.session_state["cqp_dialog_tree_leaf"] = ca
            st.rerun()
        else:
            cq = dialog_tree.add_node(dialog_tree_leaf, NodeType.CLARIFICATION_QUESTION, user_response)
            st.session_state["cqp_dialog_tree_leaf"] = cq
            st.rerun()

    if "cqp_diverse_semantic_centers" in st.session_state and st.session_state["cqp_diverse_semantic_centers"] is not None:
        with st.expander("Diverse Semantic Centers"):
            st.write(st.session_state["cqp_diverse_semantic_centers"])
    if "cqp_diverse_predictions" in st.session_state and st.session_state["cqp_diverse_predictions"] is not None:
        with st.expander("Diverse Predictions"):
            st.write(st.session_state["cqp_diverse_predictions"])
