import argparse
import math
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
        "Symptom":             "#ff9999",   # Red/Pink
        "Condition":           "#99ccff",   # Blue
        "Diagnostic_Test":     "#99ff99",   # Green
        "Vital_Sign_Threshold":"#ffcc99",   # Orange
        "Demographic_Factor":  "#cc99ff",   # Purple
        "Risk_Factor":         "#ffff99",   # Yellow
        "Mechanism_of_Injury": "#ff99cc",   # Rose
        "Clinical_Attribute":  "#99ffee",   # Mint
    }
    
    # Add nodes with their specific colors and titles (hover text)
    for node_id, node_data in G.nodes(data=True):
        node_type = node_data.get("type", "Unknown")
        # ClinGraph-sourced nodes get a distinct colour and diamond shape
        is_clingraph = node_data.get("source") == "ClinGraph"
        if is_clingraph:
            color = "#d4a0ff"   # Lavender — visually distinct from guideline nodes
            shape = "diamond"
        else:
            color = color_map.get(node_type, "#cccccc")
            shape = "dot"

        # Create a hover text string containing the metadata (e.g., SNOMED codes)
        hover_text = f"Type: {node_type}\n"
        if is_clingraph:
            hover_text += "Source: ClinGraph\n"
        for key, value in node_data.items():
            if key in ("type", "source"):
                continue
            # pd.notna() raises ValueError on lists/arrays — check type first
            if isinstance(value, list):
                if value:
                    hover_text += f"{key}: {', '.join(str(v) for v in value)}\n"
            elif pd.notna(value):
                hover_text += f"{key}: {value}\n"

        net.add_node(node_id, label=node_id, title=hover_text, color=color, shape=shape)
        
    # Add edges with relationship labels, guideline source, and evidence weights
    for source, target, edge_data in G.edges(data=True):
        relationship     = edge_data.get("relationship", "").replace("_", " ")
        guideline_source = edge_data.get("source", "Unknown")

        # Each edge type carries its own Europe PMC weight attribute:
        #   INDICATES_CONDITION → literature_weight
        #   REQUIRES_TEST       → test_literature_weight
        # Resolve whichever is present (they are mutually exclusive by edge type).
        literature_weight      = edge_data.get("literature_weight")
        test_literature_weight = edge_data.get("test_literature_weight")
        active_weight = (
            literature_weight      if (literature_weight      is not None and pd.notna(literature_weight))
            else test_literature_weight if (test_literature_weight is not None and pd.notna(test_literature_weight))
            else None
        )

        # --- Edge width: log-scaled from Europe PMC hit count ---
        # Scale: 1 hit → ~1px, 100 → ~5px, 10 000 → ~9px.
        if active_weight is not None and active_weight > 0:
            edge_width = 1 + math.log10(active_weight) * 2
        else:
            edge_width = 1

        # --- Edge label (drawn on the arrow) ---
        edge_label = f"{relationship}\n[{guideline_source}]"

        # --- Hover tooltip ---
        hover_title = f"Relationship: {relationship}\nSource: {guideline_source}"
        if active_weight is not None:
            hover_title += f"\nEurope PMC co-occurrences: {int(active_weight):,}"

        net.add_edge(source, target, title=hover_title, label=edge_label, width=edge_width)
        
    # Add physics controls so you can adjust the layout in the browser
    net.show_buttons(filter_=['physics'])
    
    # Save and output the HTML file
    output_file = "triage_knowledge_graph.html"
    net.save_graph(output_file)
    print(f"Graph saved as {output_file}. Open this file in your web browser!")

# --- Load and Visualize ---
parser = argparse.ArgumentParser(description="Visualise the triage KG as an interactive HTML.")
parser.add_argument(
    "--pkl",
    default="triage_knowledge_graph_enriched.pkl",
    help="Path to the KG pickle file (default: triage_knowledge_graph_enriched.pkl). "
         "Falls back to triage_knowledge_graph.pkl if the enriched file is not found.",
)
args = parser.parse_args()

import os
pkl_path = args.pkl
if not os.path.exists(pkl_path) and pkl_path == "triage_knowledge_graph_enriched.pkl":
    pkl_path = "triage_knowledge_graph.pkl"
    print(f"Enriched PKL not found; falling back to {pkl_path}")

print(f"Loading Knowledge Graph from {pkl_path} ...")
with open(pkl_path, 'rb') as f:
    loaded_kg = pickle.load(f)

print(f"Successfully loaded graph with {loaded_kg.number_of_nodes()} nodes.")

# Pass the loaded graph to the visualizer
visualize_interactive_kg(loaded_kg)