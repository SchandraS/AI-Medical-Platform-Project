import { Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { Layout } from "./components/Layout";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { LoginPage } from "./pages/LoginPage";
import { PredictPage } from "./pages/PredictPage";
import { BatchPage } from "./pages/BatchPage";
import { MyUploadsPage } from "./pages/MyUploadsPage";
import { ModelsPage } from "./pages/ModelsPage";
import { DashboardPage } from "./pages/DashboardPage";

function LoginRoute() {
  const { isAuthenticated } = useAuth();
  if (isAuthenticated) return <Navigate to="/dashboard" replace />;
  return <LoginPage />;
}

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<LoginRoute />} />
        <Route element={<Layout />}>
          <Route path="/dashboard" element={<ProtectedRoute><DashboardPage /></ProtectedRoute>} />
          <Route
            path="/predict"
            element={
              <ProtectedRoute minimumRole="clinician">
                <PredictPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/batch"
            element={
              <ProtectedRoute minimumRole="clinician">
                <BatchPage />
              </ProtectedRoute>
            }
          />
          <Route path="/uploads" element={<ProtectedRoute><MyUploadsPage /></ProtectedRoute>} />
          <Route path="/models" element={<ProtectedRoute><ModelsPage /></ProtectedRoute>} />
        </Route>
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </AuthProvider>
  );
}
