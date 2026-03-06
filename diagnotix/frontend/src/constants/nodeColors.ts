/** Single source of truth for node-type → hex colour.
 *  Imported by both App.tsx (sidebar legend) and GraphCanvas.tsx (canvas painter).
 */
export const NODE_COLORS: Record<string, string> = {
  Symptom:              "#E05C5C", // red
  Vital_Sign_Threshold: "#E8A838", // amber
  Demographic_Factor:   "#9B6BC4", // purple
  Risk_Factor:          "#D97B3A", // orange
  Clinical_Attribute:   "#4ABFBF", // teal
  Mechanism_of_Injury:  "#A0764A", // brown
  Condition:            "#4A90D9", // blue
  Diagnostic_Test:      "#3FB950", // green
  Treatment:            "#F0A8D0", // pink
};

export const DEFAULT_NODE_COLOR = "#8b949e";
