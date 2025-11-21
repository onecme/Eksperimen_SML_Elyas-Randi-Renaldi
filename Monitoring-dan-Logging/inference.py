import requests
import json
import joblib
import numpy as np

# ============================
# 1. LOAD SCALER
# ============================
scaler = joblib.load("scaler.pkl")

# ============================
# 2. RAW INPUT (contoh 1 siswa)
# ============================
raw_input = {
    "CGPA": 7.5,
    "Internships": 1,
    "Projects": 1,
    "Workshops/Certifications": 1,
    "AptitudeTestScore": 65,
    "SoftSkillsRating": 4.4,
    "ExtracurricularActivities": 0,  # No -> 0
    "PlacementTraining": 0,          # No -> 0
    "SSC_Marks": 61,
    "HSC_Marks": 79
}

# ============================
# 3. URUTKAN SESUAI DATA TRAINING
# ============================
column_order = [
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "ExtracurricularActivities",
    "PlacementTraining",
    "SSC_Marks",
    "HSC_Marks"
]

row = np.array([raw_input[col] for col in column_order]).reshape(1, -1)

# ============================
# 4. SCALE INPUT
# ============================
processed_row = scaler.transform(row)

# ============================
# 5. FORMAT REQUEST MLflow
# ============================
data = {
    "dataframe_split": {
        "columns": column_order,
        "data": processed_row.tolist()
    }
}

# ============================
# 6. KIRIM REQUEST
# ============================
response = requests.post(
    "http://127.0.0.1:5001/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)
)

print("\n=== RESPONSE MODEL ===")
print(response.text)
