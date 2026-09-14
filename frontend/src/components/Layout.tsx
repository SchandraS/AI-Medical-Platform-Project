import { NavLink, Outlet } from "react-router-dom";
import { useAuth, hasAtLeastRole } from "../context/AuthContext";

export function Layout() {
  const { username, role, logout } = useAuth();

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">MANAS · Diabetes Risk Console</div>
        <nav className="main-nav">
          <NavLink to="/dashboard">Dashboard</NavLink>
          {hasAtLeastRole(role, "clinician") && <NavLink to="/predict">Predict</NavLink>}
          {hasAtLeastRole(role, "clinician") && <NavLink to="/batch">Batch</NavLink>}
          <NavLink to="/uploads">My Uploads</NavLink>
          <NavLink to="/models">Models</NavLink>
        </nav>
        <div className="user-info">
          <span>
            {username} <em className="role-badge">{role}</em>
          </span>
          <button onClick={logout} className="btn-link">
            Sign out
          </button>
        </div>
      </header>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}
