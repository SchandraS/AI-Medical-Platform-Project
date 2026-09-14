import { useEffect, useState } from "react";
import { api } from "../api/client";
import { EmptyState, ErrorBanner, Spinner } from "../components/Feedback";
import type { DriftReport, InferenceLogOut, MetricsSummary, ModelVersionOut } from "../api/types";

export function DashboardPage() {
  const [active, setActive] = useState<ModelVersionOut | null>(null);
  const [activeError, setActiveError] = useState<unknown>(null);
  const [metrics, setMetrics] = useState<MetricsSummary | null>(null);
  const [drift, setDrift] = useState<DriftReport | null>(null);
  const [logs, setLogs] = useState<InferenceLogOut[] | null>(null);
  const [logsError, setLogsError] = useState<unknown>(null);

  useEffect(() => {
    api.get<ModelVersionOut>("/models/active").then(setActive).catch(setActiveError);
    api.get<MetricsSummary>("/metrics/summary?window_minutes=60").then(setMetrics).catch(() => {});
    api.get<DriftReport>("/metrics/drift?window_minutes=1440").then(setDrift).catch(() => {});
    api.get<InferenceLogOut[]>("/logs/inferences?limit=20").then(setLogs).catch(setLogsError);
  }, []);

  return (
    <div className="page dashboard-page">
      <h1>Dashboard</h1>

      <section className="dashboard-section">
        <h2>Active Model</h2>
        {activeError != null && <ErrorBanner error={activeError} />}
        {activeError == null && !active && <Spinner label="Loading active model…" />}
        {active && (
          <div className="active-model-card">
            <strong>
              {active.name} · {active.version}
            </strong>
            <span className="status-badge status-production">{active.status}</span>
            <span>{active.framework}</span>
            {active.metrics && (
              <span>
                accuracy: {(active.metrics.accuracy as number)?.toFixed(3) ?? "—"}
              </span>
            )}
          </div>
        )}
      </section>

      <section className="dashboard-section">
        <h2>Service Metrics (last hour)</h2>
        {!metrics && <Spinner label="Loading metrics…" />}
        {metrics && (
          <div className="metrics-grid">
            <div className="metric-tile">
              <span className="metric-value">{metrics.total_requests}</span>
              <span className="metric-label">Requests</span>
            </div>
            <div className="metric-tile">
              <span className="metric-value">
                {metrics.error_rate !== null ? `${(metrics.error_rate * 100).toFixed(1)}%` : "—"}
              </span>
              <span className="metric-label">Error rate</span>
            </div>
            <div className="metric-tile">
              <span className="metric-value">{metrics.latency_ms.p50?.toFixed(0) ?? "—"} ms</span>
              <span className="metric-label">p50 latency</span>
            </div>
            <div className="metric-tile">
              <span className="metric-value">{metrics.latency_ms.p95?.toFixed(0) ?? "—"} ms</span>
              <span className="metric-label">p95 latency</span>
            </div>
            <div className="metric-tile">
              <span className="metric-value">{metrics.inference_count}</span>
              <span className="metric-label">Inferences</span>
            </div>
            <div className="metric-tile">
              <span className="metric-value">
                {metrics.prediction_positive_rate !== null
                  ? `${(metrics.prediction_positive_rate * 100).toFixed(1)}%`
                  : "—"}
              </span>
              <span className="metric-label">Positive rate</span>
            </div>
          </div>
        )}
      </section>

      <section className="dashboard-section">
        <h2>Feature Drift (24h, PSI vs training baseline)</h2>
        {!drift && <Spinner label="Loading drift report…" />}
        {drift && drift.n_samples === 0 && <EmptyState>{drift.message ?? "No recent inferences."}</EmptyState>}
        {drift && drift.n_samples > 0 && (
          <table className="data-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th>PSI</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(drift.features).map(([name, d]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{d.psi.toFixed(3)}</td>
                  <td>
                    <span className={`status-badge status-${d.status}`}>{d.status.replace(/_/g, " ")}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="dashboard-section">
        <h2>Recent Inference Logs</h2>
        {logsError != null && <ErrorBanner error={logsError} />}
        {logsError == null && !logs && <Spinner label="Loading logs…" />}
        {logs && logs.length === 0 && <EmptyState>No inferences yet.</EmptyState>}
        {logs && logs.length > 0 && (
          <table className="data-table">
            <thead>
              <tr>
                <th>When</th>
                <th>Status</th>
                <th>Prediction</th>
                <th>Probability</th>
                <th>Latency</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log) => (
                <tr key={log.id} className={log.status === "error" ? "row-error" : ""}>
                  <td>{new Date(log.created_at).toLocaleString()}</td>
                  <td>{log.status}</td>
                  <td>{log.prediction ?? "—"}</td>
                  <td>{log.probability !== null ? `${(log.probability * 100).toFixed(1)}%` : "—"}</td>
                  <td>{log.latency_ms !== null ? `${log.latency_ms.toFixed(1)} ms` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
