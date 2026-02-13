import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os

st.set_page_config(layout="wide")
st.title("Dialogue Tree Visualizer")

# 1. Define the Dialogue Data
nodes_data = [
    {
        "id": "root", 
        "label": "Start", 
        "type": "root", 
        "info": "<b>User asks:</b><br>'My internet is not working.'",
        "image": "https://cdn-icons-png.flaticon.com/512/4712/4712109.png" 
    },
    {
        "id": "q1", 
        "label": "Check Lights", 
        "type": "question", 
        "info": "<b>System asks:</b><br>Are the lights on the router blinking?" 
    },
    {
        "id": "a1_yes", 
        "label": "Yes", 
        "type": "answer", 
        "info": "User answered: Yes" 
    },
    {
        "id": "a1_no", 
        "label": "No", 
        "type": "answer", 
        "info": "User answered: No" 
    },
    {
        "id": "q2_cable", 
        "label": "Check Cable", 
        "type": "question", 
        "info": "<b>System asks:</b><br>Is the power cable plugged in securely?" 
    },
    {
        "id": "q2_isp", 
        "label": "Check ISP", 
        "type": "question", 
        "info": "<b>System asks:</b><br>Is there an outage in your area?" 
    },
    {
        "id": "end_fix", 
        "label": "Solved", 
        "type": "end", 
        "info": "<b>Action:</b><br>Problem solved." 
    }
]

edges_data = [
    ("root", "q1"),
    ("q1", "a1_yes"),
    ("q1", "a1_no"),
    ("a1_yes", "q2_isp"),
    ("a1_no", "q2_cable"),
    ("q2_cable", "end_fix")
]

# 2. Build the NetworkX Graph
G = nx.DiGraph()

for node in nodes_data:
    # DEFINING NODE STYLES
    
    # Common attributes
    node_color = "#97c2fc"
    shape = "box"
    image_url = "" # PyVis prefers empty string over None for image sometimes
    
    # Custom attributes based on 'type'
    if node['type'] == 'root':
        shape = "image" 
        image_url = node['image']
        node_color = "white" 
        size = 40
    elif node['type'] == 'question':
        shape = "box"
        node_color = "#FFD700" # Gold
        size = 25
    elif node['type'] == 'answer':
        shape = "ellipse"
        node_color = "#90EE90" # Light Green
        size = 20
    elif node['type'] == 'end':
        shape = "circle"
        node_color = "#FF6961" # Red
        size = 20
    else:
        size = 20
        
    # Add node to NetworkX
    G.add_node(
        node['id'], 
        label=node['label'], 
        title=node['info'], # TOOLTIP HTML
        color=node_color,
        shape=shape,
        image=image_url,
        size=size # FIX: Always pass an integer here, never None
    )

G.add_edges_from(edges_data)

# 3. Configure PyVis
net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", directed=True)
net.from_nx(G)

# 4. Set Hierarchical Layout
net.set_options("""
var options = {
  "layout": {
    "hierarchical": {
      "enabled": true,
      "levelSeparation": 150,
      "nodeSpacing": 200,
      "treeSpacing": 200,
      "direction": "UD", 
      "sortMethod": "directed"
    }
  },
  "physics": {
    "hierarchicalRepulsion": {
      "centralGravity": 0,
      "springLength": 100,
      "springConstant": 0.01,
      "nodeDistance": 120,
      "damping": 0.09
    },
    "solver": "hierarchicalRepulsion"
  }
}
""")

# 5. Render
# This logic handles saving the HTML file in a way that works locally and on Cloud
try:
    path = '/tmp'
    net.save_graph(f'{path}/dialogue_tree.html')
    HtmlFile = open(f'{path}/dialogue_tree.html', 'r', encoding='utf-8')
except:
    # Ensure directory exists if running locally
    path = 'html_files'
    os.makedirs(path, exist_ok=True)
    net.save_graph(f'{path}/dialogue_tree.html')
    HtmlFile = open(f'{path}/dialogue_tree.html', 'r', encoding='utf-8')

components.html(HtmlFile.read(), height=600)

st.info("Hover over the nodes to see the Dialogue details.")