// Mirrors backend/app/schemas.py and app/ml/schema.py exactly.

export const FEATURE_ORDER = [
  "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
  "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
  "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
  "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
] as const;

export type FeatureName = (typeof FEATURE_ORDER)[number];

export interface FeatureSpec {
  name: FeatureName;
  label: string;
  min: number;
  max: number;
  binary: boolean;
  help?: string;
}

export const FEATURE_SPECS: FeatureSpec[] = [
  { name: "HighBP", label: "High blood pressure", min: 0, max: 1, binary: true },
  { name: "HighChol", label: "High cholesterol", min: 0, max: 1, binary: true },
  { name: "CholCheck", label: "Cholesterol check in past 5 years", min: 0, max: 1, binary: true },
  { name: "BMI", label: "Body Mass Index", min: 10, max: 100, binary: false },
  { name: "Smoker", label: "Smoked ≥100 cigarettes lifetime", min: 0, max: 1, binary: true },
  { name: "Stroke", label: "History of stroke", min: 0, max: 1, binary: true },
  { name: "HeartDiseaseorAttack", label: "CHD or heart attack history", min: 0, max: 1, binary: true },
  { name: "PhysActivity", label: "Physical activity in past 30 days", min: 0, max: 1, binary: true },
  { name: "Fruits", label: "Consumes fruit ≥1/day", min: 0, max: 1, binary: true },
  { name: "Veggies", label: "Consumes vegetables ≥1/day", min: 0, max: 1, binary: true },
  { name: "HvyAlcoholConsump", label: "Heavy alcohol consumption", min: 0, max: 1, binary: true },
  { name: "AnyHealthcare", label: "Has healthcare coverage", min: 0, max: 1, binary: true },
  { name: "NoDocbcCost", label: "Couldn't see doctor due to cost", min: 0, max: 1, binary: true },
  { name: "GenHlth", label: "Self-rated general health", min: 1, max: 5, binary: false, help: "1=excellent .. 5=poor" },
  { name: "MentHlth", label: "Days mental health not good (30d)", min: 0, max: 30, binary: false },
  { name: "PhysHlth", label: "Days physical health not good (30d)", min: 0, max: 30, binary: false },
  { name: "DiffWalk", label: "Difficulty walking/climbing stairs", min: 0, max: 1, binary: true },
  { name: "Sex", label: "Sex", min: 0, max: 1, binary: true, help: "0=female, 1=male" },
  { name: "Age", label: "Age category", min: 1, max: 13, binary: false, help: "1=18-24 .. 13=80+" },
  { name: "Education", label: "Education level", min: 1, max: 6, binary: false },
  { name: "Income", label: "Income level", min: 1, max: 8, binary: false },
];

export type PatientRecord = Record<FeatureName, number>;

export interface PredictResponse {
  prediction: number;
  label: string;
  probability: number;
  confidence: number;
  threshold: number;
  model_version_id: string;
  model_name: string;
  model_version: string;
  preprocessing_version: string;
  input_hash: string;
  input_echo: Record<string, number>;
  timestamp: string;
  latency_ms: number;
  log_id: string;
}

export interface ApiError {
  detail: string;
  error_code?: string | null;
  errors?: { loc: string[]; msg: string; type: string }[];
  request_id?: string | null;
}

export interface BatchJobAccepted {
  job_id: string;
  status: string;
  total_rows: number;
  message: string;
}

export interface BatchJobStatus {
  job_id: string;
  filename: string;
  status: "queued" | "running" | "succeeded" | "failed" | "partial";
  total_rows: number;
  processed_rows: number;
  succeeded_rows: number;
  failed_rows: number;
  model_version_id: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface BatchRowResult {
  row_index: number;
  status: "ok" | "error";
  prediction: number | null;
  label: string | null;
  probability: number | null;
  confidence: number | null;
  error: string | null;
}

export interface BatchJobResults {
  job_id: string;
  status: string;
  rows: BatchRowResult[];
}

export interface ModelVersionOut {
  id: string;
  name: string;
  version: string;
  framework: string;
  status: "candidate" | "staging" | "production" | "archived";
  metrics: Record<string, number | null> | null;
  artifact_sha256: string;
  preprocessing_version: string;
  created_at: string;
  promoted_at: string | null;
}

export interface PromotionEventOut {
  id: string;
  model_version_id: string;
  from_status: string | null;
  to_status: string;
  actor_user_id: string | null;
  reason: string | null;
  metrics_snapshot: Record<string, number | null> | null;
  created_at: string;
}

export interface InferenceLogOut {
  id: string;
  user_id: string | null;
  batch_job_id: string | null;
  row_index: number | null;
  model_version_id: string | null;
  input_hash: string;
  prediction: number | null;
  probability: number | null;
  status: string;
  error_code: string | null;
  latency_ms: number | null;
  created_at: string;
}

export interface MetricsSummary {
  window_minutes: number;
  total_requests: number;
  error_requests: number;
  error_rate: number | null;
  latency_ms: { p50: number | null; p95: number | null; p99: number | null };
  inference_count: number;
  inference_error_count: number;
  prediction_positive_rate: number | null;
  active_model: { id: string; name: string; version: string } | null;
}

export interface DriftReport {
  window_minutes: number;
  n_samples: number;
  features: Record<string, { psi: number; status: string }>;
  overall_status?: string;
  message?: string;
}

export type Role = "viewer" | "clinician" | "ml_engineer";
