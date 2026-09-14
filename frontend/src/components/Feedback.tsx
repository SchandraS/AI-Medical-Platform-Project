import { ApiRequestError } from "../api/client";

export function Spinner({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="feedback feedback-loading">
      <span className="spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

export function ErrorBanner({ error }: { error: unknown }) {
  if (!error) return null;

  let detail = "Something went wrong.";
  let fieldErrors: { loc: string[]; msg: string }[] | undefined;

  if (error instanceof ApiRequestError) {
    detail = error.body.detail;
    fieldErrors = error.body.errors;
  } else if (error instanceof Error) {
    detail = error.message;
  }

  return (
    <div className="feedback feedback-error" role="alert">
      <strong>Error:</strong> {detail}
      {fieldErrors && fieldErrors.length > 0 && (
        <ul>
          {fieldErrors.map((e, i) => (
            <li key={i}>
              <code>{e.loc.join(".")}</code>: {e.msg}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function EmptyState({ children }: { children: React.ReactNode }) {
  return <div className="feedback feedback-empty">{children}</div>;
}
