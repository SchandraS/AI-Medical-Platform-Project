import { Navigate } from "react-router-dom";
import { useAuth, hasAtLeastRole } from "../context/AuthContext";
import type { Role } from "../api/types";

export function ProtectedRoute({
  children,
  minimumRole = "viewer",
}: {
  children: React.ReactNode;
  minimumRole?: Role;
}) {
  const { isAuthenticated, role } = useAuth();

  if (!isAuthenticated) return <Navigate to="/login" replace />;
  if (!hasAtLeastRole(role, minimumRole)) {
    return (
      <div className="feedback feedback-error">
        You need the "{minimumRole}" role or higher to view this page. You are signed in as "{role}".
      </div>
    );
  }
  return <>{children}</>;
}
