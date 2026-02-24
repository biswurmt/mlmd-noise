
# 🧠 Medical Triage Knowledge Graph

This repository contains the pipeline to build and visualize a custom medical knowledge graph for Emergency Department triage. It incorporates clinical guidelines from ACR, AHA/ACC, NICE, and CTAS to map symptoms, vitals, and demographics to specific diagnostic tests.

## ⚙️ Prerequisites

Ensure you have installed the required dependencies before running the scripts:

```bash
pip install -r requirements.txt
```

## 🔑 Environment Setup

To query the proprietary medical ontologies (SNOMED CT US and CA), you must configure your API credentials.

Create a file named `.env` in the root directory of this project and add your specific keys:

```text
UMLS_API_KEY=your_real_long_umls_key_here
INFOWAY_TOKEN=your_real_infoway_token_here
```

## 🚀 Usage

### Step 1: Build the Knowledge Graph

Run the build script to extract the clinical guidelines, query the medical APIs for standardized ontology codes, and construct the network.

```bash
python build_kg.py
```

* **Output:** This generates a `triage_knowledge_graph.pkl` file, saving the fully compiled NetworkX graph locally.

### Step 2: Visualize the Knowledge Graph

Run the visualization script to load the saved graph and create an interactive web interface.

```bash
python visualize_kg.py
```

* **Output:** This generates a `triage_knowledge_graph.html` file. Open this file in any web browser to drag, zoom, and explore the nodes, as well as view the specific guideline sources attached to each relationship edge.