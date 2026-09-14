"""The single source of truth for the 21-feature clinical input contract.

This exact name/order was extracted from scaler.feature_names_in_ (the fitted
StandardScaler shipped with the model) — NOT guessed from the Kaggle page.
Ranges below were empirically verified against the real BRFSS2015 5050-split
CSV (min/max per column). Every binary field is a strict {0, 1} indicator.

Do not reorder this list. app.ml.predictor builds the model input by reading
this list's order; the RandomForest has no feature_names_in_ of its own, so
this is the only thing standing between "predict" and a silently-wrong
clinical answer with wrong-order features.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    description: str
    minimum: float
    maximum: float
    binary: bool = False


# Order matches scaler.feature_names_in_ exactly.
FEATURES: list[FeatureSpec] = [
    FeatureSpec("HighBP", "High blood pressure", 0, 1, binary=True),
    FeatureSpec("HighChol", "High cholesterol", 0, 1, binary=True),
    FeatureSpec("CholCheck", "Cholesterol check in past 5 years", 0, 1, binary=True),
    FeatureSpec("BMI", "Body Mass Index", 10, 100),
    FeatureSpec("Smoker", "Smoked >=100 cigarettes lifetime", 0, 1, binary=True),
    FeatureSpec("Stroke", "History of stroke", 0, 1, binary=True),
    FeatureSpec("HeartDiseaseorAttack", "CHD or MI history", 0, 1, binary=True),
    FeatureSpec("PhysActivity", "Physical activity in past 30 days", 0, 1, binary=True),
    FeatureSpec("Fruits", "Consumes fruit >=1/day", 0, 1, binary=True),
    FeatureSpec("Veggies", "Consumes vegetables >=1/day", 0, 1, binary=True),
    FeatureSpec("HvyAlcoholConsump", "Heavy alcohol consumption", 0, 1, binary=True),
    FeatureSpec("AnyHealthcare", "Has any healthcare coverage", 0, 1, binary=True),
    FeatureSpec("NoDocbcCost", "Could not see doctor due to cost", 0, 1, binary=True),
    FeatureSpec("GenHlth", "Self-rated general health (1=excellent..5=poor)", 1, 5),
    FeatureSpec("MentHlth", "Days mental health not good (past 30 days)", 0, 30),
    FeatureSpec("PhysHlth", "Days physical health not good (past 30 days)", 0, 30),
    FeatureSpec("DiffWalk", "Difficulty walking/climbing stairs", 0, 1, binary=True),
    FeatureSpec("Sex", "Sex (0=female, 1=male)", 0, 1, binary=True),
    FeatureSpec("Age", "Age category (1=18-24 .. 13=80+)", 1, 13),
    FeatureSpec("Education", "Education level (1..6)", 1, 6),
    FeatureSpec("Income", "Income level (1..8)", 1, 8),
]

FEATURE_ORDER: list[str] = [f.name for f in FEATURES]
FEATURE_MAP: dict[str, FeatureSpec] = {f.name: f for f in FEATURES}
TARGET_COLUMN = "Diabetes_binary"

CLASS_LABELS = {0: "no_diabetes", 1: "diabetes_or_prediabetes"}
