"""
delete_nodes.py — Edit nodes in triage_knowledge_graph_enriched.pkl

Three modes of operation:

  1. Delete a specific node by its full ID (any type):
       python delete_nodes.py --node "Symptom: Pediatric Assessment"
       python delete_nodes.py --node "Condition: Appendicitis"

  2. Delete an entire test pathway and its exclusively-owned nodes:
       python delete_nodes.py "ECG"
       python delete_nodes.py "Testicular Ultrasound"
       python delete_nodes.py          # interactive menu

  3. Prune all nodes with no undirected path to any Diagnostic_Test:
       python delete_nodes.py --clean

  Other:
       python delete_nodes.py --list          # list all Test nodes
       python delete_nodes.py --list-all      # list every node, grouped by type
       python delete_nodes.py --node "..." --dry-run  # preview without saving
       python delete_nodes.py --clean --dry-run

A timestamped backup PKL is always created before any write (unless --dry-run).
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


def clean_dangling(G, dry_run: bool = False):
    """Remove every node that cannot reach any Diagnostic_Test node via
    directed paths, including fully isolated nodes (zero edges).

    Uses directed ancestor traversal rather than undirected reachability so
    that isolated nodes (degree 0) and nodes that only connect sideways — not
    toward a test — are all caught correctly.
    """
    test_nodes = set(get_test_nodes(G))
    if not test_nodes:
        print("No Diagnostic_Test nodes found — nothing to clean.")
        return set()

    # Collect every node that lies on a directed path TO a test node.
    # nx.ancestors(G, t) = all nodes from which t is reachable via directed edges.
    reachable: set = set()
    for t in test_nodes:
        reachable.update(nx.ancestors(G, t))
        reachable.add(t)  # include the test node itself

    dangling = set(G.nodes) - reachable

    if not dangling:
        print("No dangling nodes found — graph is already clean.")
        return set()

    # Group output by reason for clarity
    zero_edge = sorted(n for n in dangling if G.degree(n) == 0)
    has_edges  = sorted(n for n in dangling if G.degree(n) > 0)

    print(f"Found {len(dangling)} dangling node(s) with no directed path to any test:\n")

    if zero_edge:
        print(f"  Isolated (0 edges) — {len(zero_edge)}:")
        for n in zero_edge:
            t = G.nodes[n].get("type", "—")
            print(f"    [-] {n}  [{t}]")

    if has_edges:
        print(f"\n  Has edges but no test connection — {len(has_edges)}:")
        for n in has_edges:
            t     = G.nodes[n].get("type", "—")
            in_e  = G.in_degree(n)
            out_e = G.out_degree(n)
            print(f"    [-] {n}  [{t}]  ({in_e} in / {out_e} out)")

    print()

    if dry_run:
        print(f"[dry-run] Would delete {len(dangling)} node(s). No changes made.")
        return dangling

    confirm = input(f"Delete all {len(dangling)} dangling node(s)? [y/N]: ").strip().lower()
    if confirm == "y":
        G.remove_nodes_from(dangling)
        save_graph(G)
        print(f"\nDone. Graph now has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    else:
        print("Aborted — no changes made.")

    return dangling


def delete_specific_node(G, node_id: str, dry_run: bool = False):
    """Delete a single node by its exact node_id string.

    Removes all incident edges automatically (NetworkX behaviour).
    Prints a preview of what will be removed, then prompts for confirmation
    unless dry_run is True (in which case no changes are made).
    """
    if node_id not in G:
        # Fuzzy hint: find nodes whose label contains the query
        query = node_id.lower()
        matches = [n for n in G.nodes if query in str(n).lower()]
        print(f"[ERROR] Node not found: '{node_id}'")
        if matches:
            print(f"\nDid you mean one of these ({len(matches)} match(es))?")
            for m in sorted(matches)[:20]:
                t = G.nodes[m].get("type", "—")
                print(f"  {m}  [{t}]")
        raise SystemExit(1)

    node_type  = G.nodes[node_id].get("type", "—")
    in_edges   = list(G.in_edges(node_id))
    out_edges  = list(G.out_edges(node_id))

    print(f"\nNode to DELETE: '{node_id}'  [{node_type}]")
    print(f"  Incoming edges ({len(in_edges)}):")
    for src, _ in in_edges:
        print(f"    {src}  →  {node_id}")
    print(f"  Outgoing edges ({len(out_edges)}):")
    for _, tgt in out_edges:
        print(f"    {node_id}  →  {tgt}")
    print()

    if dry_run:
        print("[dry-run] No changes made.")
        return

    confirm = input("Confirm deletion? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted — no changes made.")
        return

    G.remove_node(node_id)
    save_graph(G)
    print(f"\nDone. Graph now has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")


def list_tests(G):
    tests = get_test_nodes(G)
    print(f"{len(tests)} Diagnostic_Test node(s):\n")
    for t in tests:
        preds = list(G.predecessors(t))
        src = G.nodes[t].get("guideline_source", "—")
        print(f"  {t}  (source: {src}, {len(preds)} incoming node(s))")


def list_all_nodes(G):
    """Print every node grouped by type, with edge counts."""
    from collections import defaultdict
    by_type: dict[str, list] = defaultdict(list)
    for n, d in G.nodes(data=True):
        by_type[d.get("type", "—")].append(str(n))

    total = G.number_of_nodes()
    print(f"{total} nodes across {len(by_type)} type(s):\n")
    for ntype in sorted(by_type):
        nodes = sorted(by_type[ntype])
        print(f"── {ntype} ({len(nodes)}) ──")
        for n in nodes:
            in_e  = G.in_degree(n)
            out_e = G.out_degree(n)
            print(f"  {n}  ({in_e} in / {out_e} out)")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Edit nodes in triage_knowledge_graph_enriched.pkl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "test", nargs="?",
        help="Test label to delete as a whole pathway, e.g. 'ECG'",
    )
    parser.add_argument(
        "--node", metavar="NODE_ID",
        help="Delete a specific node by its full ID, e.g. 'Symptom: Pediatric Assessment'",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all Diagnostic_Test nodes and exit",
    )
    parser.add_argument(
        "--list-all", action="store_true",
        help="List every node in the graph grouped by type and exit",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove all nodes with no undirected path to any Diagnostic_Test node",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without modifying the PKL (no backup written)",
    )
    args = parser.parse_args()

    G = load_graph()
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    if args.list:
        list_tests(G)
    elif args.list_all:
        list_all_nodes(G)
    elif args.node:
        delete_specific_node(G, args.node, dry_run=args.dry_run)
    elif args.clean:
        clean_dangling(G, dry_run=args.dry_run)
    elif args.test:
        delete_by_name(G, args.test)
    else:
        interactive_mode(G)


if __name__ == "__main__":
    main()
