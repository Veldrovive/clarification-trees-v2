import streamlit as st

from clarification_trees.dialog_tree import DialogTrajectory, NodeType

def dialog_trajectory_component(dialog_trajectory: DialogTrajectory, allow_text_input: bool = False, allow_generate_input: bool = False, assistant_name: str = "Assistant"):
    """
    Renders a dialog trajectory in a streamlit app.
    """
    with st.container(border=False, width="content"):
        next_user_name = "User"
        for node in dialog_trajectory.trajectory[::-1]:
            if node.node_type == NodeType.CLARIFICATION_QUESTION:
                user_name = assistant_name
                next_user_name = "User"
            elif node.node_type == NodeType.CLARIFYING_ANSWER:
                user_name = "User"
                next_user_name = assistant_name
            elif node.node_type == NodeType.ROOT:
                user_name = "User"
                next_user_name = assistant_name
            elif node.node_type == NodeType.INFERENCE:
                user_name = assistant_name
                next_user_name = "User"
            else:
                raise ValueError(f"Unknown node type: {node.node_type}")
            
            with st.chat_message(user_name):
                if node.image is not None:
                    st.image(node.image)
                    st.write(node.response)
                else:
                    st.write(node.response)

        if allow_text_input and allow_generate_input:
            with st.chat_message(next_user_name):
                cols = st.columns([2, 1])
                with cols[1]:
                    if st.button("Generate"):
                        return None, True
                with cols[0]:
                    with st.form("dialog_traj_form", clear_on_submit=True):
                        input = st.text_input("Your Answer", key="dialog_traj_input")
                        if st.form_submit_button("Submit"):
                            if input is not None and input.strip() != "":
                                return input, False
        elif allow_text_input:
            with st.chat_message(next_user_name):
                with st.form("dialog_traj_form", clear_on_submit=True):
                    input = st.text_input("Your Answer", key="dialog_traj_input")
                    if st.form_submit_button("Submit"):
                        if input is not None and input.strip() != "":
                            return input, False
        elif allow_generate_input:
            with st.chat_message(next_user_name):
                if st.button("Generate"):
                    return None, True
        
        return None, False