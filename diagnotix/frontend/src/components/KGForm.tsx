import { useState, FormEvent } from "react";

interface Props {
  onSubmit: (diagnosticTest: string) => void;
}

const EXAMPLES = [
  "CT Head",
  "Chest X-Ray",
  "MRI Spine",
  "Lumbar Puncture",
  "Echocardiogram",
  "Abdominal CT",
];

export default function KGForm({ onSubmit }: Props) {
  const [value, setValue] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = value.trim();
    if (trimmed) onSubmit(trimmed);
  }

  return (
    <div className="form-card">
      <div className="form-header">
        <h2>Add a Diagnostic Test</h2>
        <p className="form-description">
          Enter any diagnostic test to automatically extract clinical triage
          rules using AI, then ground every node with SNOMED CT, ICD-10, LOINC
          codes, and evidence weights from Europe PMC and ClinicalTrials.gov.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="form">
        <div className="input-group">
          <label htmlFor="diagnostic-test" className="input-label">
            Diagnostic Test
          </label>
          <input
            id="diagnostic-test"
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder='e.g. "CT Head", "Chest X-Ray", "MRI Spine"'
            className="text-input"
            autoFocus
            autoComplete="off"
          />
        </div>

        <button
          type="submit"
          className="btn-submit"
          disabled={!value.trim()}
        >
          Extract Guidelines and Build Graph
          <span className="btn-arrow">→</span>
        </button>
      </form>

      <div className="examples">
        <p className="examples-label">Quick examples:</p>
        <div className="examples-list">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              className="example-chip"
              onClick={() => setValue(ex)}
              type="button"
            >
              {ex}
            </button>
          ))}
        </div>
      </div>

      <div className="pipeline-info">
        <p className="pipeline-title">What happens when you submit:</p>
        <ol className="pipeline-steps">
          <li>Claude AI reads clinical guidelines and generates triage rules</li>
          <li>Rules are appended to <code>guideline_rules.json</code></li>
          <li>
            Ontology grounding runs: EBI OLS4, Infoway SNOMED CT, LOINC,
            ICD-10-CM, RxNorm, OpenFDA
          </li>
          <li>
            Evidence weights fetched from Europe PMC and ClinicalTrials.gov
          </li>
          <li>
            NetworkX graph rebuilt and saved to{" "}
            <code>triage_knowledge_graph.pkl</code>
          </li>
        </ol>
      </div>
    </div>
  );
}
