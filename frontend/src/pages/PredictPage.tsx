import { useState } from "react";
import { api, ApiRequestError } from "../api/client";
import { ErrorBanner, Spinner } from "../components/Feedback";
import { FEATURE_SPECS, type PatientRecord, type PredictResponse } from "../api/types";

const DEFAULT_RECORD: PatientRecord = Object.fromEntries(
  FEATURE_SPECS.map((f) => [f.name, f.binary ? 0 : f.min])
) as PatientRecord;

type InputMode = "form" | "csv";

export function PredictPage() {
  const [mode, setMode] = useState<InputMode>("form");
  const [record, setRecord] = useState<PatientRecord>({ ...DEFAULT_RECORD });
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  function updateField(name: string, value: number) {
    setRecord((r) => ({ ...r, [name]: value }));
  }

  function parseFieldErrors(err: unknown) {
    if (err instanceof ApiRequestError && err.body.errors) {
      const fe: Record<string, string> = {};
      for (const fieldErr of err.body.errors) {
        const key = fieldErr.loc[fieldErr.loc.length - 1];
        fe[key] = fieldErr.msg;
      }
      setFieldErrors(fe);
    }
  }

  async function handleFormSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setFieldErrors({});
    setResult(null);
    try {
      const resp = await api.post<PredictResponse>("/predict", { record });
      setResult(resp);
    } catch (err) {
      setError(err);
      parseFieldErrors(err);
    } finally {
      setLoading(false);
    }
  }

  async function handleCsvSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!csvFile) return;
    setLoading(true);
    setError(null);
    setFieldErrors({});
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", csvFile);
      const resp = await api.postForm<PredictResponse>("/predict/csv", form);
      setResult(resp);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page predict-page">
      <h1>Single Prediction</h1>
      <p className="page-subtitle">
        Submit a single patient record — fill the form, or upload a one-row CSV with a header matching the 21
        fields. All fields are required; values outside the valid range are rejected.
      </p>

      <div className="mode-toggle" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={mode === "form"}
          className={mode === "form" ? "tab-active" : "tab-inactive"}
          onClick={() => setMode("form")}
        >
          Form
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "csv"}
          className={mode === "csv" ? "tab-active" : "tab-inactive"}
          onClick={() => setMode("csv")}
        >
          Upload CSV
        </button>
      </div>

      <div className="predict-layout">
        {mode === "form" ? (
          <form onSubmit={handleFormSubmit} className="predict-form">
            <div className="field-grid">
              {FEATURE_SPECS.map((spec) => (
                <label key={spec.name} className={fieldErrors[spec.name] ? "field-invalid" : ""}>
                  <span>
                    {spec.label}
                    {spec.help && <small> ({spec.help})</small>}
                  </span>
                  {spec.binary ? (
                    <select
                      value={record[spec.name]}
                      onChange={(e) => updateField(spec.name, Number(e.target.value))}
                    >
                      <option value={0}>0 — No</option>
                      <option value={1}>1 — Yes</option>
                    </select>
                  ) : (
                    <input
                      type="number"
                      value={record[spec.name]}
                      min={spec.min}
                      max={spec.max}
                      step="any"
                      onChange={(e) => updateField(spec.name, Number(e.target.value))}
                    />
                  )}
                  {fieldErrors[spec.name] && <small className="field-error">{fieldErrors[spec.name]}</small>}
                </label>
              ))}
            </div>

            <button type="submit" disabled={loading}>
              {loading ? "Running inference…" : "Predict"}
            </button>
          </form>
        ) : (
          <form onSubmit={handleCsvSubmit} className="predict-form">
            <p>
              Upload a CSV with a header row of the 21 feature names (HighBP, HighChol, BMI, …) and exactly one
              data row.
            </p>
            <input type="file" accept=".csv" onChange={(e) => setCsvFile(e.target.files?.[0] ?? null)} />
            <button type="submit" disabled={!csvFile || loading}>
              {loading ? "Running inference…" : "Predict from CSV"}
            </button>
          </form>
        )}

        <div className="predict-result">
          {loading && <Spinner label="Running inference…" />}
          {!loading && error != null && <ErrorBanner error={error} />}
          {!loading && error == null && !result && (
            <div className="feedback feedback-empty">
              {mode === "form"
                ? "Fill the form and click Predict to see a result."
                : "Choose a one-row CSV and click Predict from CSV to see a result."}
            </div>
          )}
          {result && (
            <div className={`result-card result-${result.prediction === 1 ? "positive" : "negative"}`}>
              <h2>{result.label.replace(/_/g, " ")}</h2>
              <dl>
                <dt>Prediction</dt>
                <dd>{result.prediction}</dd>
                <dt>Probability</dt>
                <dd>{(result.probability * 100).toFixed(1)}%</dd>
                <dt>Confidence</dt>
                <dd>{(result.confidence * 100).toFixed(1)}%</dd>
                <dt>Model</dt>
                <dd>
                  {result.model_name} · {result.model_version}
                </dd>
                <dt>Preprocessing version</dt>
                <dd>{result.preprocessing_version}</dd>
                <dt>Input hash</dt>
                <dd>
                  <code>{result.input_hash.slice(0, 16)}…</code>
                </dd>
                <dt>Latency</dt>
                <dd>{result.latency_ms.toFixed(1)} ms</dd>
                <dt>Timestamp</dt>
                <dd>{new Date(result.timestamp).toLocaleString()}</dd>
              </dl>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
