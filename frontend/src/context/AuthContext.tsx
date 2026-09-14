import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { api, ApiRequestError, setToken } from "../api/client";
import type { Role } from "../api/types";

interface AuthState {
  username: string | null;
  role: Role | null;
  isAuthenticated: boolean;
}

interface LoginResult {
  access_token: string;
  role: Role;
  username: string;
}

interface AuthContextValue extends AuthState {
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  loading: boolean;
  error: string | null;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const ROLE_RANK: Record<Role, number> = { viewer: 0, clinician: 1, ml_engineer: 2 };

export function hasAtLeastRole(role: Role | null, minimum: Role): boolean {
  if (!role) return false;
  return ROLE_RANK[role] >= ROLE_RANK[minimum];
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [username, setUsername] = useState<string | null>(() => localStorage.getItem("manas_username"));
  const [role, setRole] = useState<Role | null>(() => (localStorage.getItem("manas_role") as Role) || null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const login = useCallback(async (u: string, p: string) => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.post<LoginResult>("/auth/login", { username: u, password: p });
      setToken(result.access_token);
      localStorage.setItem("manas_username", result.username);
      localStorage.setItem("manas_role", result.role);
      setUsername(result.username);
      setRole(result.role);
    } catch (err) {
      if (err instanceof ApiRequestError) {
        setError(err.body.detail);
      } else {
        setError("Login failed unexpectedly.");
      }
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    setToken(null);
    localStorage.removeItem("manas_username");
    localStorage.removeItem("manas_role");
    setUsername(null);
    setRole(null);
  }, []);

  const value = useMemo(
    () => ({ username, role, isAuthenticated: !!username, login, logout, loading, error }),
    [username, role, login, logout, loading, error]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
