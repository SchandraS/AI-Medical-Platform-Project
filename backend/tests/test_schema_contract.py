"""Guards against the single highest-risk regression in this codebase: the
Pydantic validation ranges (app.schemas.PatientRecord) silently drifting from
the ML feature contract (app.ml.schema.FEATURE_ORDER / FEATURE_MAP), which
were extracted directly from scaler.feature_names_in_. If these ever
diverge, predictions could be silently wrong or fields silently misvalidated.
"""
from app.ml.schema import FEATURE_MAP, FEATURE_ORDER
from app.schemas import PatientRecord


def test_patient_record_has_exactly_the_expected_fields():
    assert set(PatientRecord.model_fields.keys()) == set(FEATURE_ORDER)


def test_patient_record_ranges_match_feature_spec():
    for name, spec in FEATURE_MAP.items():
        field = PatientRecord.model_fields[name]
        constraints = {m.__class__.__name__: m for m in field.metadata}
        ge = next((m.ge for m in field.metadata if hasattr(m, "ge") and m.ge is not None), None)
        le = next((m.le for m in field.metadata if hasattr(m, "le") and m.le is not None), None)
        assert ge == spec.minimum, f"{name}: expected min {spec.minimum}, got {ge}"
        assert le == spec.maximum, f"{name}: expected max {spec.maximum}, got {le}"


def test_to_feature_dict_preserves_order_and_all_fields():
    record = PatientRecord(**{name: (0 if FEATURE_MAP[name].binary else FEATURE_MAP[name].minimum) for name in FEATURE_ORDER})
    d = record.to_feature_dict()
    assert list(d.keys()) == FEATURE_ORDER
