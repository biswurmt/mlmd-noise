import { type FormEvent, useState } from "react";

const EXAMPLES = [
  "CT Head",
  "Chest X-Ray",
  "Troponin",
];

interface Props {
  onSubmit: (diagnosticTest: string) => void;
  disabled: boolean;
}

export default function ChatInput({ onSubmit, disabled }: Props) {
  const [value, setValue] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setValue("");
  }

  return (
    <div className="chat-bar">
      <div className="chat-examples">
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            className="example-chip"
            disabled={disabled}
            onClick={() => {
              setValue(ex);
            }}
          >
            {ex}
          </button>
        ))}
      </div>

      <form className="chat-form" onSubmit={handleSubmit}>
        <input
          className="chat-input"
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Enter a diagnostic test (e.g. ECG, CT Abdomen)…"
          disabled={disabled}
          autoComplete="off"
          spellCheck={false}
        />
        <button className="chat-submit" type="submit" disabled={disabled || !value.trim()}>
          {disabled ? (
            <span className="spinner-sm" />
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          )}
        </button>
      </form>
    </div>
  );
}
