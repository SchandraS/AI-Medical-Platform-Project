import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useAuth, hasAtLeastRole } from "../context/AuthContext";
import { EmptyState, ErrorBanner, Spinner } from "../components/Feedback";
import type { ModelVersionOut } from "../api/types";

export function ModelsPage() {
  const { role } = useAuth();
  const isMlEngineer = hasAtLeastRole(role, "ml_engineer");

  const [versions, setVersions] = useState<ModelVersionOut[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<unknown>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  function reload() {
    setError(null);
    api.get<ModelVersionOut[]>("/models").then(setVersions).catch(setError);
  }

  useEffect(reload, []);

  async function handleEvaluate(id: string) {
    setBusyId(id);
    setActionError(null);
    setActionMessage(null);
    try {
      await api.post(`/models/${id}/evaluate`);
      setActionMessage("Evaluation complete — metrics updated below.");
      reload();
    } catch (err) {
      setActionError(err);
    } finally {
      setBusyId(null);
    }
  }

  async function handlePromote(id: string) {
    const reason = window.prompt("Reason for promoting this version to production:");
    if (!reason) return;
    setBusyId(id);
    setActionError(null);
    setActionMessage(null);
    try {
      await api.post(`/models/${id}/promote`, { reason });
      setActionMessage("Promoted to production.");
      reload();
    } catch (err) {
      setActionError(err);
    } finally {
      setBusyId(null);
    }
  }

  async function handleRollback() {
    const reason = window.prompt("Reason for rolling back the current production model:");
    if (!reason) return;
    setBusyId("rollback");
    setActionError(null);
    setActionMessage(null);
    try {
      await api.post("/models/rollback", { reason });
      setActionMessage("Rolled back to the previous production version.");
      reload();
    } catch (err) {
      setActionError(err);
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="page">
      <h1>Model Registry</h1>
      <p className="page-subtitle">
        Every registered model version, its lifecycle status, and evaluation metrics. Promotion requires the
        ml_engineer role and always demotes the current production version to "archived" rather than overwriting it.
      </p>

      {isMlEngineer && (
        <button onClick={handleRollback} disabled={busyId === "rollback"} className="btn-danger">
          {busyId === "rollback" ? "Rolling back…" : "Rollback production to previous version"}
        </button>
      )}
      {actionMessage && <div className="feedback feedback-success">{actionMessage}</div>}
      {actionError != null && <ErrorBanner error={actionError} />}

      {error != null && <ErrorBanner error={error} />}
      {error == null && versions === null && <Spinner label="Loading model versions…" />}
      {versions !== null && versions.length === 0 && <EmptyState>No model versions registered.</EmptyState>}

      {versions && versions.length > 0 && (
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Version</th>
              <th>Framework</th>
              <th>Status</th>
              <th>Metrics</th>
              <th>Promoted</th>
              {isMlEngineer && <th>Actions</th>}
            </tr>
          </thead>
          <tbody>
            {versions.map((v) => (
              <tr key={v.id}>
                <td>{v.name}</td>
                <td>{v.version}</td>
                <td>{v.framework}</td>
                <td>
                  <span className={`status-badge status-${v.status}`}>{v.status}</span>
                </td>
                <td>
                  {v.metrics ? (
                    <ul className="metrics-list">
                      {Object.entries(v.metrics).map(([k, val]) => (
                        <li key={k}>
                          {k}: {typeof val === "number" ? val.toFixed(3) : String(val)}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    "—"
                  )}
                </td>
                <td>{v.promoted_at ? new Date(v.promoted_at).toLocaleString() : "—"}</td>
                {isMlEngineer && (
                  <td className="actions-cell">
                    <button disabled={busyId === v.id} onClick={() => handleEvaluate(v.id)}>
                      Evaluate
                    </button>
                    {v.status !== "production" && (
                      <button disabled={busyId === v.id} onClick={() => handlePromote(v.id)}>
                        Promote
                      </button>
                    )}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
