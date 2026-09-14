# Take-Home Assignment

## MANAS Platform, MLOps and Clinical AI Deployment

---

### **Submission Deadline: September 15, 2026, 08:00 PM (IST)**

---

### What You're Given

- You can get your pretrained model form the: [https://huggingface.co/Neperl/diabetes-health-indicators-study](https://huggingface.co/Neperl/diabetes-health-indicators-study)  
- Treat the two models provided in this repo as two versions of the prediction model.  
- For getting the data you can use any one of the binary sub-classification dataset from: [https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data)

---

### The Task

Convert the supplied model(s) into a small, production-style inference service.

> Preferred stack: FastAPI (backend), React (frontend), PostgreSQL (persistence \- for logging inferences, model metadata, versions, etc.). You are free to substitute any part of this stack with something you're more comfortable with (e.g., Flask/Django, Vue/Svelte, SQLite/MongoDB), as long as you can justify the choice and it doesn't compromise the requirements below. If you deviate, note it briefly in your README.

#### 1\. Inference Service (REST API)

- Implement a REST API (FastAPI preferred, but any reasonable Python framework is acceptable).  
- The service should:  
  - Accept patient/sample data (single record) as CSV/JSON.  
  - Validate the input (correct fields, types, ranges \- no silent coercion of garbage data).  
  - Run inference and return:  
    - The predicted class/label  
    - A confidence/probability score  
    - Metadata sufficient to reproduce the inference later (e.g., model version used, timestamp, input hash/echo, preprocessing version).  
  - Support both single-request and batch inference.  
  - All relevant data should be stored persistently so that it can be retrieved and viewed later.  
  - Create automated unit and integration tests using pytest covering, at minimum: valid single and batch inference, invalid/malformed input, missing or unexpected fields, input validation and range checks, model/version loading and selection, model inference failures, persistence of inference results and logs, and API error handling.

#### 2\. Frontend

- Build a small frontend (React preferred) that talks to your backend service. It doesn't need to be visually polished, but it should be functional and should demonstrate that your API is usable by a real client, not just curl/Postman. At minimum it should let a user:  
  - Submit a single sample (a form, or uploading a csv) and see the prediction, confidence, and model version used.  
  - Upload and trigger batch inference on similar data.  
  - See which model version is currently active/deployed, and ideally a basic view of recent inference logs/history.  
  - The user can see all his uploaded csv and its results(batched/single).  
  - A minimal dashboard for viewing recent predictions/model status.  
- Keep the frontend simple and scoped \- we're evaluating that you can wire a real client to your API cleanly (loading states, error states, sane display of validation errors from the backend), not your UI design skills.

#### 3\. Containerization

- Provide a Dockerfile for the API, a Dockerfile for the frontend, and a docker-compose.yml that wires them together along with your database (e.g., a postgres service, or your chosen equivalent). Running docker-compose up (or equivalent) should bring up a working end-to-end system: frontend talking to the backend, backend talking to the database.

#### 4\. Model Versioning & Logging

- Design a simple but sound mechanism for:  
  - Model versioning \- multiple model files must coexist; the service should be able to specify/select which version is active.  
  - Logging \- every inference request/response (and errors) should be logged with enough metadata to trace back what model, what inputs, and what output were involved.  
- Persist logs, model version metadata, and (ideally) promotion history in a proper database rather than flat files \- PostgreSQL is our preferred choice, but SQLite or another store is acceptable for a take-home as long as the schema/design is sound and you explain the tradeoff. This is what your React app's "recent inference logs" and "active model version" views should be reading from.  
- Explain (in your README) how a new model version would be evaluated and promoted into production without silently replacing the currently deployed model. We expect at least a description of:  
  - Where new models "land" before serving traffic (e.g., staged/candidate state).  
  - What validation/evaluation happens before promotion.  
  - How promotion and rollback actually happen operationally.

#### 5\. Robustness

Your service must handle the following gracefully (i.e., no unhandled 500s, no crashes, clear error responses):

- Malformed input (bad CSV, wrong types, extra/unexpected fields).  
- Missing required features.  
- Model load/inference failure (e.g., corrupted model file, incompatible input shape).  
- Automated tests should also verify that the service returns clear, appropriate HTTP error responses rather than unhandled 500 errors for expected failure scenarios.

---

### Bonus (Not Mandatory)

Any of the following will strengthen your submission but are not required to pass:

- RBAC Authentication System  
- Basic monitoring (latency, error rate, prediction distribution drift).  
- API documentation (OpenAPI/Swagger, or a well-written README with example calls).

---

### Question (Required)

You receive 1,000 new labelled samples after deployment. Would your deployed model automatically retrain itself? Design the update process.

Please answer this in a separate DESIGN.md. We're specifically interested in whether your design includes:

- A frozen production model that is not silently mutated by new data.  
- A separate retraining process (offline, isolated from the live service).  
- A validation/challenger comparison step (new model vs. current production model on held-out/eval data).  
- An explicit approval step before any promotion.  
- Versioned deployment of the new model (not an in-place overwrite).  
- A rollback path if the new version underperforms in production.

We are specifically not looking for a design where the deployed model continuously and automatically retrains itself on incoming data without human/process gates \- in clinical/health contexts this is a safety concern, not just an engineering one.  
---

### Deliverables

Submit a GitHub repository with the following structure:

repo-root/  
├── backend/           \# Full API setup (source code, requirements/deps, etc.) \+ Dockerfile  
│   └── Dockerfile  
├── frontend/           \# Full frontend setup (source code, package config, etc.) \+ Dockerfile  
│   └── Dockerfile  
├── docker-compose.yml   \# One command to build images/containers and run the full Web-App  
└── README.md               \# Explains how you to setup and run the Web-App  
└── REPORT.md               \# Write a short report about all of your choice not exceeding 5 pages  
└── DESIGN.md                \# Answer to the question asked above

1. backend/ \- the full API setup, including its own Dockerfile to build the backend container.  
2. frontend/ \- the full frontend setup, including its own Dockerfile to build the frontend container.  
3. REPORT.md — Explain how you approached and implemented each part of the assignment, including:  
   - Example single and batch API requests and responses.  
   - Testing approach, including key test cases and results.  
4. REPORT.md \- explain how you approached and solved each part of the problem, and call out anything else worth knowing about your choices. At minimum, cover:  
   - Example requests (single \+ batch) and responses.  
   - Your versioning and logging design.  
   - Your promotion/rollback strategy for new model versions.  
   - Any assumptions, design decisions, tradeoffs, or limitations, with a brief explanation of why they were chosen.  
5. [DESIGN.md](http://DESIGN.md) \- Answer to the question asked above  
6. docker-compose.yml (at repo root) \- running this should build the images/containers (backend, frontend, database) and bring up the full working web app in one step.  
7. Testing \- Include automated tests using pytest and document in README.md how to install test dependencies, run the test suite, and view the test results.  
8. Any tests, docs, or additional dashboard/monitoring you choose to include (bonus) \- place these logically within backend/ or frontend/, or in a top-level docs/ folder if repo-wide