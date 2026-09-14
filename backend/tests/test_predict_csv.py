"""Single-record prediction via CSV upload (/predict/csv) — the CSV
counterpart to /predict's JSON body, per the assignment's requirement that
a single sample be submittable "as CSV/JSON" and, on the frontend, via "a
form, or uploading a csv"."""
import csv
import io

from tests.conftest import VALID_RECORD, auth_headers


def _csv_bytes(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def test_predict_csv_valid_single_row_returns_full_metadata(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    csv_bytes = _csv_bytes([VALID_RECORD])
    resp = client.post(
        "/predict/csv", headers=headers, files={"file": ("sample.csv", csv_bytes, "text/csv")}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version_id"] == production_model_version.id
    assert body["input_echo"]["BMI"] == VALID_RECORD["BMI"]
    assert body["log_id"]


def test_predict_csv_persists_inference_log(client, seeded_users, production_model_version, db_session):
    from app.models_db import InferenceLog

    headers = auth_headers(client, "clinician", "clinician123")
    csv_bytes = _csv_bytes([VALID_RECORD])
    resp = client.post(
        "/predict/csv", headers=headers, files={"file": ("sample.csv", csv_bytes, "text/csv")}
    )
    assert resp.status_code == 200
    log = db_session.query(InferenceLog).filter(InferenceLog.id == resp.json()["log_id"]).first()
    assert log is not None
    assert log.status == "ok"
    assert log.model_version_id == production_model_version.id


def test_predict_csv_matches_json_predict_for_same_record(client, seeded_users, production_model_version):
    """The CSV and JSON single-predict paths must be equivalent for the
    same input — proving they share the same validation/inference logic."""
    headers = auth_headers(client, "clinician", "clinician123")

    json_resp = client.post("/predict", json={"record": VALID_RECORD}, headers=headers)
    csv_resp = client.post(
        "/predict/csv", headers=headers,
        files={"file": ("sample.csv", _csv_bytes([VALID_RECORD]), "text/csv")},
    )
    assert json_resp.status_code == csv_resp.status_code == 200
    assert json_resp.json()["prediction"] == csv_resp.json()["prediction"]
    assert json_resp.json()["probability"] == csv_resp.json()["probability"]
    assert json_resp.json()["input_hash"] == csv_resp.json()["input_hash"]


def test_predict_csv_rejects_multiple_rows(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    csv_bytes = _csv_bytes([VALID_RECORD, VALID_RECORD])
    resp = client.post(
        "/predict/csv", headers=headers, files={"file": ("sample.csv", csv_bytes, "text/csv")}
    )
    assert resp.status_code == 422
    assert "exactly one data row" in resp.json()["detail"]


def test_predict_csv_rejects_empty_file(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.post(
        "/predict/csv", headers=headers, files={"file": ("sample.csv", b"", "text/csv")}
    )
    assert resp.status_code == 422


def test_predict_csv_rejects_non_csv_extension(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.post(
        "/predict/csv", headers=headers, files={"file": ("sample.txt", b"not a csv", "text/plain")}
    )
    assert resp.status_code == 422


def test_predict_csv_rejects_header_only_csv(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    header_only = (",".join(VALID_RECORD.keys()) + "\n").encode()
    resp = client.post(
        "/predict/csv", headers=headers, files={"file": ("sample.csv", header_only, "text/csv")}
    )
    assert resp.status_code == 422


def test_predict_csv_rejects_missing_field(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    bad = {k: v for k, v in VALID_RECORD.items() if k != "BMI"}
    resp = client.post(
        "/predict/csv", headers=headers, files={"file": ("sample.csv", _csv_bytes([bad]), "text/csv")}
    )
    assert resp.status_code == 422


def test_predict_csv_rejects_out_of_range_value(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    bad = {**VALID_RECORD, "BMI": 500}
    resp = client.post(
        "/predict/csv", headers=headers, files={"file": ("sample.csv", _csv_bytes([bad]), "text/csv")}
    )
    assert resp.status_code == 422


def test_predict_csv_rejects_extra_column(client, seeded_users, production_model_version):
    headers = auth_headers(client, "clinician", "clinician123")
    bad = {**VALID_RECORD, "SomeExtraColumn": 1}
    resp = client.post(
        "/predict/csv", headers=headers, files={"file": ("sample.csv", _csv_bytes([bad]), "text/csv")}
    )
    assert resp.status_code == 422


def test_predict_csv_requires_auth(client, seeded_users, production_model_version):
    resp = client.post(
        "/predict/csv", files={"file": ("sample.csv", _csv_bytes([VALID_RECORD]), "text/csv")}
    )
    assert resp.status_code == 401


def test_predict_csv_viewer_role_forbidden(client, seeded_users, production_model_version):
    headers = auth_headers(client, "viewer", "viewer123")
    resp = client.post(
        "/predict/csv", headers=headers, files={"file": ("sample.csv", _csv_bytes([VALID_RECORD]), "text/csv")}
    )
    assert resp.status_code == 403


def test_predict_csv_no_production_model_returns_503(client, seeded_users):
    headers = auth_headers(client, "clinician", "clinician123")
    resp = client.post(
        "/predict/csv", headers=headers, files={"file": ("sample.csv", _csv_bytes([VALID_RECORD]), "text/csv")}
    )
    assert resp.status_code == 503
