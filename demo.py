"""
Uses streamlit to demo the components of the clarification tree system.
"""
import dotenv
dotenv.load_dotenv()

import streamlit as st
from omegaconf import DictConfig, OmegaConf
import hydra

from clarification_trees.streamlit.pages import clarifying_question_page

st.set_page_config(layout="wide", page_title="Clarification Tree Demo")

def initialize_session_state(cfg: DictConfig):
    if "initialized" in st.session_state:
        return

    st.session_state["cfg"] = cfg
    st.session_state["current_page"] = None

    st.session_state["initialized"] = True

@hydra.main(config_path="src/clarification_trees/config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    initialize_session_state(cfg)

    st.title("Clarification Tree Demo")

    with st.container(horizontal_alignment="center"):
        clarifying_question_page()

    # dataset = load_clearvqa_dataset(cfg)
    
    # test_sample = dataset[1]
    # test_img = test_sample[0]
    # blurred_question = test_sample[1]["blurred_question"]
    # clarifying_question = test_sample[1]["clarification_question"]

    # tree = DialogTree(blurred_question, test_img)
    # cq = tree.add_node(DialogTree.ROOT, NodeType.CLARIFICATION_QUESTION, None, clarifying_question)
    # trajectory = tree.get_trajectory(cq)

    # render_dialog_trajectory(trajectory)
    # answer = st.text_input("Answer", value="")
    # if answer:
    #     ca = tree.add_node(cq, NodeType.CLARIFYING_ANSWER, None, answer)
    #     trajectory = tree.get_trajectory(ca)
    #     render_dialog_trajectory(trajectory)
    
    

    

    
if __name__ == "__main__":
    main()