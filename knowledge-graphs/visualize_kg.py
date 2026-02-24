import pickle
import pandas as pd
from pyvis.network import Network
import networkx as nx

def visualize_interactive_kg(G):
    print("Generating interactive graph...")
    
    # Initialize a PyVis network
    net = Network(notebook=False, directed=True, height="800px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # Color palette for node types
    color_map = {
        "Symptom": "#ff9999",               # Red/Pink
        "Condition": "#99ccff",             # Blue
        "Diagnostic_Test": "#99ff99",       # Green
        "Vital_Sign_Threshold": "#ffcc99",  # Orange
        "Demographic_Factor": "#cc99ff"     # Purple
    }
    
    # Add nodes with their specific colors and titles (hover text)
    for node_id, node_data in G.nodes(data=True):
        node_type = node_data.get("type", "Unknown")
        color = color_map.get(node_type, "#cccccc")
        
        # Create a hover text string containing the metadata (e.g., SNOMED codes)
        hover_text = f"Type: {node_type}\n"
        for key, value in node_data.items():
            if key != "type" and pd.notna(value):
                hover_text += f"{key}: {value}\n"
                
        net.add_node(node_id, label=node_id, title=hover_text, color=color, shape="dot")
        
    # Add edges with relationship labels
    # --- NEW: Add edges with relationship labels AND source ---
    for source, target, edge_data in G.edges(data=True):
        relationship = edge_data.get("relationship", "").replace("_", " ")
        guideline_source = edge_data.get("source", "Unknown")
        
        # The label appears directly on the drawn arrow
        edge_label = f"{relationship}\n[{guideline_source}]"
        
        # The title creates a tooltip when you hover your mouse over the arrow
        hover_title = f"Relationship: {relationship}\nSource: {guideline_source}"
        
        net.add_edge(source, target, title=hover_title, label=edge_label)
        
    # Add physics controls so you can adjust the layout in the browser
    net.show_buttons(filter_=['physics'])
    
    # Save and output the HTML file
    output_file = "triage_knowledge_graph.html"
    net.save_graph(output_file)
    print(f"Graph saved as {output_file}. Open this file in your web browser!")

# --- Load and Visualize ---
print("Loading Knowledge Graph...")
with open('triage_knowledge_graph.pkl', 'rb') as f:
    loaded_kg = pickle.load(f)

print(f"Successfully loaded graph with {loaded_kg.number_of_nodes()} nodes.")

# Pass the loaded graph to the visualizer
visualize_interactive_kg(loaded_kg)