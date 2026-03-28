"""
Filter the epfl-llm/guidelines dataset to rows relevant to our 5 diagnostic tests.
Saves filtered subsets to knowledge-graphs/data/guidelines_filtered/.
"""
from datasets import load_from_disk
import re, json, os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

KEYWORDS = {
    "ecg": [
        r"\becg\b", r"\bekg\b", r"\belectrocardiogr", r"\bcardiac monitor",
        r"\bacute coronary", r"\bacs\b", r"\bstemi\b", r"\bnstemi\b",
        r"\bmyocardial infarct", r"\bangina\b", r"\barrhythmia\b",
    ],
    "testicular_ultrasound": [
        r"\btesticular ultrasound", r"\bscrotal ultrasound", r"\bscrotal us\b",
        r"\btesticular torsion", r"\bscrotal pain", r"\bepididymitis\b",
        r"\borchitis\b", r"\btesticular pain", r"\bhydrocele\b",
    ],
    "arm_xray": [
        r"\barm (x-?ray|fracture|injur)", r"\bwrist (x-?ray|fracture)",
        r"\bforearm fracture", r"\bradius fracture", r"\bulna fracture",
        r"\bfoosh\b", r"\bcolles", r"\bdistal radius",
        r"\bupper (limb|extremity) fracture", r"\bhumeral fracture",
    ],
    "appendix_ultrasound": [
        r"\bappendicit", r"\bappendix ultrasound", r"\bappendic(eal|ular)",
        r"\brlq pain", r"\bright lower quadrant",
        r"\bappendix\b.*\bimagin", r"\bimagin.*\bappendix",
    ],
    "abdominal_ultrasound": [
        r"\babdominal ultrasound", r"\babdominal us\b", r"\babdominal sonograph",
        r"\babdominal imaging", r"\bupper abdomen.*ultrasound",
        r"\bhepatic ultrasound", r"\bgallbladder ultrasound",
        r"\brenal ultrasound", r"\baortic ultrasound",
    ],
}


def main():
    print("Loading dataset...")
    train = load_from_disk(os.path.join(DATA_DIR, "guidelines"))["train"]
    print(f"Total rows: {len(train)}")

    compiled = {
        test: [re.compile(p, re.IGNORECASE) for p in pats]
        for test, pats in KEYWORDS.items()
    }

    # Tag each row with which tests it matches
    matched_indices = {t: [] for t in KEYWORDS}

    for i, row in enumerate(train):
        text = (row.get("title") or "") + " " + (row.get("clean_text") or "")
        for test, patterns in compiled.items():
            if any(p.search(text) for p in patterns):
                matched_indices[test].append(i)

    print("\n=== Match counts ===")
    for test, idxs in matched_indices.items():
        print(f"  {test:30s}: {len(idxs):5d} rows")

    # Save each filtered subset
    out_dir = os.path.join(DATA_DIR, "guidelines_filtered")
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    for test, idxs in matched_indices.items():
        subset = train.select(idxs)
        # Add a label column
        subset = subset.map(lambda _: {"diagnostic_test": test}, desc=f"labelling {test}")
        path = os.path.join(out_dir, test)
        subset.save_to_disk(path)
        summary[test] = {"count": len(idxs), "path": path}
        print(f"Saved {len(idxs)} rows → {path}")

    # Also save a combined dataset with all matches (deduped by id, with tags)
    all_indices = sorted(set(i for idxs in matched_indices.values() for i in idxs))

    def tag_row(row, idx):
        text = (row.get("title") or "") + " " + (row.get("clean_text") or "")
        tags = [t for t, pats in compiled.items() if any(p.search(text) for p in pats)]
        return {"diagnostic_tests": ", ".join(tags)}

    combined = train.select(all_indices)
    combined = combined.map(tag_row, with_indices=True, desc="tagging combined")
    combined_path = os.path.join(out_dir, "combined")
    combined.save_to_disk(combined_path)
    print(f"\nSaved combined ({len(all_indices)} unique rows) → {combined_path}")

    # Write a summary JSON
    summary["combined"] = {"count": len(all_indices), "path": combined_path}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
