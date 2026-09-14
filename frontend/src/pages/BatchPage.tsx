import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiRequestError } from "../api/client";
import { ErrorBanner, Spinner } from "../components/Feedback";
import type { BatchJobAccepted, BatchJobStatus } from "../api/types";

const POLL_INTERVAL_MS = 2000;
const TERMINAL_STATUSES = new Set(["succeeded", "failed", "partial"]);

export function BatchPage() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<unknown>(null);
  const [activeJob, setActiveJob] = useState<BatchJobStatus | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const navigate = useNavigate();

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  function pollJob(jobId: string) {
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const status = await api.get<BatchJobStatus>(`/jobs/${jobId}`);
        setActiveJob(status);
        if (TERMINAL_STATUSES.has(status.status)) stopPolling();
      } catch {
        stopPolling();
      }
    }, POLL_INTERVAL_MS);
  }

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    setActiveJob(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const accepted = await api.postForm<BatchJobAccepted>("/jobs/upload", form);
      const status = await api.get<BatchJobStatus>(`/jobs/${accepted.job_id}`);
      setActiveJob(status);
      pollJob(accepted.job_id);
    } catch (err) {
      setUploadError(err);
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="page">
      <h1>Batch Inference</h1>
      <p className="page-subtitle">
        Upload a CSV of patient records (header row matching the 21 fields, e.g. HighBP, HighChol, BMI, …). Rows are
        validated and scored individually — bad rows don't block good ones.
      </p>

      <form onSubmit={handleUpload} className="upload-form">
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button type="submit" disabled={!file || uploading}>
          {uploading ? "Uploading…" : "Upload & Run Batch"}
        </button>
      </form>

      {uploading && <Spinner label="Uploading and queuing job…" />}
      {uploadError != null && <ErrorBanner error={uploadError} />}

      {activeJob && (
        <div className="job-status-card">
          <h2>Job {activeJob.job_id.slice(0, 8)}…</h2>
          <p>
            File: <strong>{activeJob.filename}</strong>
          </p>
          <p>
            Status: <span className={`status-badge status-${activeJob.status}`}>{activeJob.status}</span>
          </p>
          <progress value={activeJob.processed_rows} max={activeJob.total_rows || 1} />
          <p>
            {activeJob.processed_rows} / {activeJob.total_rows} rows processed
            {" · "}
            {activeJob.succeeded_rows} succeeded, {activeJob.failed_rows} failed
          </p>
          {TERMINAL_STATUSES.has(activeJob.status) && (
            <button onClick={() => navigate(`/uploads?job=${activeJob.job_id}`)}>View row-level results</button>
          )}
        </div>
      )}
    </div>
  );
}
