"""
delete_nodes.py — Remove a diagnostic test pathway from triage_knowledge_graph.pkl

Deletes the chosen Test node plus any Condition and primary nodes (Symptom,
Vital, Demographic, Risk Factor, Attribute, MOI) that become fully orphaned
after the test is removed (i.e. they had no edges to any other Test node).
Nodes shared with other tests are preserved; only their edges to the deleted
test are removed.

Usage:
  # Interactive — pick a test from the menu
  python delete_nodes.py

  # Delete a specific test by exact name
  python delete_nodes.py "ECG"
  python delete_nodes.py "Testicular Ultrasound"

  # List all Test nodes and exit
  python delete_nodes.py --list

  # Remove all nodes/edges not reachable from any Test node
  python delete_nodes.py --clean
"""

import argparse
import os
import pickle
import shutil
from datetime import datetime

import networkx as nx

PKL_PATH = os.path.join(os.path.dirname(__file__), "triage_knowledge_graph_enriched.pkl")

TEST_TYPE = "Diagnostic_Test"


def load_graph():
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)


def save_graph(G):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = PKL_PATH.replace(".pkl", f"_backup_{ts}.pkl")
    shutil.copy2(PKL_PATH, backup)
    print(f"Backup saved → {os.path.basename(backup)}")
    with open(PKL_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved  → {os.path.basename(PKL_PATH)}")


def get_test_nodes(G):
    return sorted(
        n for n, d in G.nodes(data=True) if d.get("type") == TEST_TYPE
    )


def resolve_test_node(G, name):
    """Accept a bare label ('ECG') or a full node ID ('Test: ECG')."""
    if name in G and G.nodes[name].get("type") == TEST_TYPE:
        return name
    candidate = f"Test: {name}"
    if candidate in G and G.nodes[candidate].get("type") == TEST_TYPE:
        return candidate
    return None


def compute_deletion_set(G, test_node):
    """
    Return (nodes_to_delete, edges_to_remove_only) for a test-level deletion.

    nodes_to_delete   — test node + any conditions/primaries exclusively tied to it
    edges_to_remove_only — edges from shared nodes that pointed to this test
                           (the shared nodes themselves are kept)
    """
    other_tests = set(get_test_nodes(G)) - {test_node}

    def connected_to_other_test(node):
        """True if node has any path to a test other than the one being deleted."""
        for _, tgt in G.out_edges(node):
            if tgt in other_tests:
                return True
            if G.nodes[tgt].get("type") == TEST_TYPE:
                return True  # guard for any future test nodes
        return False

    to_delete = {test_node}
    edges_only = []  # (src, tgt) edges to remove from shared nodes

    # Walk predecessors of the test node
    for pred in list(G.predecessors(test_node)):
        pred_type = G.nodes[pred].get("type", "")

        if pred_type == "Condition":
            # Check if this condition points to any other test
            if connected_to_other_test(pred):
                edges_only.append((pred, test_node))
                # Also check condition's own predecessors (primary nodes)
                for ppred in G.predecessors(pred):
                    if not connected_to_other_test(ppred):
                        # Primary node only connected through this condition
                        # but condition is shared — just remove direct edge to test
                        edges_only.append((ppred, test_node))
            else:
                to_delete.add(pred)
                # Cascade to primary nodes that fed this condition
                for ppred in G.predecessors(pred):
                    if connected_to_other_test(ppred):
                        edges_only.append((ppred, test_node))
                    else:
                        to_delete.add(ppred)

        else:
            # Primary node pointing directly to the test
            if connected_to_other_test(pred):
                edges_only.append((pred, test_node))
            else:
                to_delete.add(pred)

    return to_delete, edges_only


def print_plan(G, test_node, to_delete, edges_only):
    print(f"\nTest pathway: {test_node}")
    print(f"  Source: {G.nodes[test_node].get('guideline_source', '—')}")
    print()

    exclusively = sorted(to_delete - {test_node})
    print(f"  Nodes to DELETE ({len(exclusively)} + the test itself):")
    for n in exclusively:
        t = G.nodes[n].get("type", "")
        print(f"    [-] {n}  [{t}]")

    if edges_only:
        print(f"\n  Edges to REMOVE from shared nodes ({len(edges_only)}):")
        for src, tgt in edges_only:
            print(f"    [-] {src}  →  {tgt}")

    print()


def apply_deletion(G, to_delete, edges_only):
    for src, tgt in edges_only:
        if G.has_edge(src, tgt):
            G.remove_edge(src, tgt)
    G.remove_nodes_from(to_delete)


def interactive_mode(G):
    tests = get_test_nodes(G)
    if not tests:
        print("No Diagnostic_Test nodes found in graph.")
        return

    print("=== Diagnostic Test pathways ===\n")
    for i, t in enumerate(tests):
        src = G.nodes[t].get("guideline_source", "—")
        print(f"  [{i}] {t}  (source: {src})")

    sel = input("\nSelect test index to delete (or 'quit'): ").strip()
    if sel.lower() == "quit":
        print("Aborted.")
        return

    if not sel.isdigit() or int(sel) >= len(tests):
        print("Invalid selection.")
        return

    test_node = tests[int(sel)]
    to_delete, edges_only = compute_deletion_set(G, test_node)
    print_plan(G, test_node, to_delete, edges_only)

    confirm = input("Confirm deletion? [y/N]: ").strip().lower()
    if confirm == "y":
        apply_deletion(G, to_delete, edges_only)
        save_graph(G)
        print(f"\nDone. Graph now has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    else:
        print("Aborted — no changes made.")


def delete_by_name(G, name):
    test_node = resolve_test_node(G, name)
    if not test_node:
        print(f"No Diagnostic_Test node found for '{name}'.")
        print("Available tests:")
        for t in get_test_nodes(G):
            print(f"  {t}")
        raise SystemExit(1)

    to_delete, edges_only = compute_deletion_set(G, test_node)
    print_plan(G, test_node, to_delete, edges_only)

    confirm = input("Confirm deletion? [y/N]: ").strip().lower()
    if confirm == "y":
        apply_deletion(G, to_delete, edges_only)
        save_graph(G)
        print(f"\nDone. Graph now has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    else:
        print("Aborted — no changes made.")


def clean_dangling(G):
    """
    Remove every node (and its edges) that has no undirected path to any
    Diagnostic_Test node.  Returns the set of removed node IDs.
    """
    test_nodes = set(get_test_nodes(G))
    if not test_nodes:
        print("No Diagnostic_Test nodes found — nothing to clean.")
        return set()

    # Build undirected view once for reachability checks
    U = G.to_undirected()
    connected = set()
    for t in test_nodes:
        connected.update(nx.node_connected_component(U, t))

    dangling = set(G.nodes) - connected

    if not dangling:
        print("No dangling nodes found — graph is already clean.")
        return set()

    print(f"Found {len(dangling)} dangling node(s) with no path to any test:\n")
    for n in sorted(dangling):
        t = G.nodes[n].get("type", "—")
        in_e  = G.in_degree(n)
        out_e = G.out_degree(n)
        print(f"  [-] {n}  [{t}]  ({in_e} in / {out_e} out edges)")

    confirm = input(f"\nDelete all {len(dangling)} dangling node(s)? [y/N]: ").strip().lower()
    if confirm == "y":
        G.remove_nodes_from(dangling)
        save_graph(G)
        print(f"\nDone. Graph now has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    else:
        print("Aborted — no changes made.")

    return dangling


def list_tests(G):
    tests = get_test_nodes(G)
    print(f"{len(tests)} Diagnostic_Test node(s):\n")
    for t in tests:
        preds = list(G.predecessors(t))
        src = G.nodes[t].get("guideline_source", "—")
        print(f"  {t}  (source: {src}, {len(preds)} incoming node(s))")


def main():
    parser = argparse.ArgumentParser(
        description="Delete a diagnostic test pathway from triage_knowledge_graph.pkl"
    )
    parser.add_argument(
        "test", nargs="?",
        help="Test label to delete, e.g. 'ECG' or 'Testicular Ultrasound'"
    )
    parser.add_argument("--list",  action="store_true", help="List all tests and exit")
    parser.add_argument("--clean", action="store_true",
                        help="Remove all nodes not reachable from any Test node")
    args = parser.parse_args()

    G = load_graph()
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    if args.list:
        list_tests(G)
    elif args.clean:
        clean_dangling(G)
    elif args.test:
        delete_by_name(G, args.test)
    else:
        interactive_mode(G)


if __name__ == "__main__":
    main()
