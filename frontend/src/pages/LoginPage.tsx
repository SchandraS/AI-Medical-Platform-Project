import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const DEMO_ACCOUNTS = [
  { username: "clinician", password: "clinician123", role: "Clinician — run predictions & batches" },
  { username: "mlops", password: "mlops123", role: "ML Engineer — full access incl. promotion" },
  { username: "viewer", password: "viewer123", role: "Viewer — read-only dashboard" },
];

export function LoginPage() {
  const { login, loading, error } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    try {
      await login(username, password);
      navigate("/dashboard");
    } catch {
      // error is surfaced via useAuth().error
    }
  }

  function fillDemo(u: string, p: string) {
    setUsername(u);
    setPassword(p);
  }

  return (
    <div className="login-page">
      <form className="login-card" onSubmit={handleSubmit}>
        <h1>MANAS</h1>
        <p className="subtitle">Diabetes Risk Inference Console</p>

        <label>
          Username
          <input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus required />
        </label>
        <label>
          Password
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </label>

        {error && <div className="feedback feedback-error">{error}</div>}

        <button type="submit" disabled={loading}>
          {loading ? "Signing in…" : "Sign in"}
        </button>

        <div className="demo-accounts">
          <p>Demo accounts (seeded at startup):</p>
          <ul>
            {DEMO_ACCOUNTS.map((acc) => (
              <li key={acc.username}>
                <button type="button" className="btn-link" onClick={() => fillDemo(acc.username, acc.password)}>
                  {acc.username} / {acc.password}
                </button>{" "}
                — {acc.role}
              </li>
            ))}
          </ul>
        </div>
      </form>
    </div>
  );
}
