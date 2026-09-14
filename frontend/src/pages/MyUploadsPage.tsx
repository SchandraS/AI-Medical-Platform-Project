import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { EmptyState, ErrorBanner, Spinner } from "../components/Feedback";
import type { BatchJobResults, BatchJobStatus } from "../api/types";

export function MyUploadsPage() {
  const [jobs, setJobs] = useState<BatchJobStatus[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedJobId = searchParams.get("job");
  const [results, setResults] = useState<BatchJobResults | null>(null);
  const [resultsError, setResultsError] = useState<unknown>(null);
  const [resultsLoading, setResultsLoading] = useState(false);

  useEffect(() => {
    api
      .get<BatchJobStatus[]>("/jobs")
      .then(setJobs)
      .catch(setError);
  }, []);

  useEffect(() => {
    if (!selectedJobId) {
      setResults(null);
      return;
    }
    setResultsLoading(true);
    setResultsError(null);
    api
      .get<BatchJobResults>(`/jobs/${selectedJobId}/results`)
      .then(setResults)
      .catch(setResultsError)
      .finally(() => setResultsLoading(false));
  }, [selectedJobId]);

  return (
    <div className="page">
      <h1>My Uploads</h1>
      <p className="page-subtitle">Every CSV you've uploaded and its batch inference outcome.</p>

      {error != null && <ErrorBanner error={error} />}
      {error == null && jobs === null && <Spinner label="Loading uploads…" />}
      {jobs !== null && jobs.length === 0 && <EmptyState>No uploads yet — try the Batch page.</EmptyState>}

      {jobs && jobs.length > 0 && (
        <table className="data-table">
          <thead>
            <tr>
              <th>File</th>
              <th>Status</th>
              <th>Rows</th>
              <th>Succeeded</th>
              <th>Failed</th>
              <th>Uploaded</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.job_id} className={job.job_id === selectedJobId ? "row-selected" : ""}>
                <td>{job.filename}</td>
                <td>
                  <span className={`status-badge status-${job.status}`}>{job.status}</span>
                </td>
                <td>{job.total_rows}</td>
                <td>{job.succeeded_rows}</td>
                <td>{job.failed_rows}</td>
                <td>{new Date(job.created_at).toLocaleString()}</td>
                <td>
                  <button className="btn-link" onClick={() => setSearchParams({ job: job.job_id })}>
                    View results
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selectedJobId && (
        <div className="job-results">
          <h2>Row-level results</h2>
          {resultsLoading && <Spinner label="Loading results…" />}
          {resultsError != null && <ErrorBanner error={resultsError} />}
          {results && results.rows.length === 0 && <EmptyState>No rows recorded yet.</EmptyState>}
          {results && results.rows.length > 0 && (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Row</th>
                  <th>Status</th>
                  <th>Prediction</th>
                  <th>Probability</th>
                  <th>Error</th>
                </tr>
              </thead>
              <tbody>
                {results.rows.map((row) => (
                  <tr key={row.row_index} className={row.status === "error" ? "row-error" : ""}>
                    <td>{row.row_index}</td>
                    <td>{row.status}</td>
                    <td>{row.prediction ?? "—"}</td>
                    <td>{row.probability !== null ? `${(row.probability * 100).toFixed(1)}%` : "—"}</td>
                    <td>{row.error ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
