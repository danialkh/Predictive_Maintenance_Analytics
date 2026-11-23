# Vehicle Predictive Maintenance Analytics

Integrated final project for the Master's Degree in Computer Science and Engineering (**Intelligent Embedded Systems** major) at **Alma Mater Studiorum – University of Bologna** (A.Y. 2025/2026, November 2026):

- **Machine Learning & Data Mining (Codice 95531)** — Predictive modeling, leakage-free evaluation, Isolation Forest outlier analysis, and WEKA comparison.
- **Distributed Systems (Codice 87474)** — Blockchain-backed VIN digital twins, smart contracts, and immutable service records (Hyperledger Fabric).
- **Embedded Systems & IoT (Codice 77780)** — ESP32 telemetry acquisition via OBD-II, local edge pre-filtering, and MQTT reporting.

**Authors:** Danial Khayatian, Dias Katrenov, Mary Anne Selirio  
**Supervisor:** Prof. Matteo Golfarelli

---

## Repository Contents

- `Predictive_Maintenance.ipynb` — End-to-end Python pipeline (EDA, feature engineering, stratified split, Isolation Forest analysis, scikit-learn pipelines, cross-validation, test evaluation, learning curves, permutation importance, decision tree rule inspection, and cost-benefit analysis).
- `Predictive_Maintenance_Analytics.pdf` — Full project report.
- `Datasets/` — Raw dataset (`vehicle_maintenance_data.csv`, 50,000 rows × 20 columns) and WEKA format (`data.arff`).
- `weka/` — WEKA experiment outputs (`Adaboost-decietionStrump.txt`, `functions-logistic.txt`, `trees-randomForest.txt`) and experiment reports.

---

## Key Findings & Results

### 1. Data Understanding & Deterministic Rule
- The raw dataset contains **50,000 records** and **20 columns** (19 features + binary target `Need_Maintenance`: `40,498` positive / `81.0%`, `9,502` negative / `19.0%`), with zero missing values or exact duplicate rows.
- Exploratory analysis indicates a likely synthetic dataset:
  - `18,096` records (`36.2%`) have `Mileage` greater than `Odometer_Reading`.
  - Grouping by the six primary driver attributes (`Battery_Status`, `Brake_Condition`, `Reported_Issues`, `Maintenance_History`, `Service_History`, `Accident_History`) yields **6,479 unique combinations** with **zero label conflicts**, revealing a deterministic rule behind `Need_Maintenance`.

### 2. Leakage-Free Methodology & Outlier Analysis
- Date columns (`Last_Service_Date`, `Warranty_Expiry_Date`) are deterministically converted into `Days_Since_Service` and `Days_To_Warranty_End` relative to `2024-03-01` before splitting.
- A **stratified 80/20 split** (`40,000` train / `10,000` test) is performed prior to data-driven preprocessing, with scaling and one-hot encoding fitted strictly inside `ColumnTransformer` pipelines on training folds only.
- An **Isolation Forest** (`200` trees, `contamination=0.01`) fitted on training features flags `400` records (`1.0%`). Inspection shows they represent rare-combination but valid vehicles rather than corrupted entries, and **stratified 5-fold cross-validation** confirms removing them yields no performance gain. All `40,000` training records are retained.

### 3. Test-Set Performance & Cost-Benefit Analysis (10,000 Vehicles)
Using illustrative error costs of **€500 per False Negative** (missed maintenance) and **€50 per False Positive** (unnecessary inspection):

| Model | Accuracy | ROC-AUC | TN / FP / FN / TP | Error Cost | Savings vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (`class_weight='balanced'`) | `96.0%` | `0.988` | `1,893` / `7` / `393` / `7,707` | **€196,850** | Baseline |
| **Random Forest** (`100` trees) | `100.0%` | `1.000` | `1,900` / `0` / `0` / `8,100` | **€0** | **€196,850** (`€19.68` / vehicle) |

*(Note: Because the dataset appears synthetic with a deterministic label rule, 100% accuracy and illustrative savings should not be expected directly on noisy real-world fleet telemetry.)*

### 4. WEKA Comparison (10,000 Test Instances)
- **AdaBoost (Decision Stumps):** `96.96%` accuracy (`Kappa = 0.8996`, `MCC = 0.900`).
- **Logistic Regression:** `96.02%` accuracy (`Kappa = 0.8680`, `ROC Area = 0.993`).
- **Random Forest (`-depth 10`):** `86.78%` accuracy (`Kappa = 0.3901`), illustrating how tree depth constraints affect rule capture.

### 5. Integrated Architecture (ML + Blockchain + IoT)
- **ML Service:** Flask `/predict` endpoint serving the Random Forest pipeline for maintenance probability/risk classification alongside Isolation Forest anomaly flagging.
- **Blockchain Layer:** Hyperledger Fabric VIN-based digital twin and smart contracts (e.g., triggering maintenance workflows when predicted probability exceeds `80%`) for tamper-evident service history.
- **IoT Layer:** ESP32 nodes reading OBD-II diagnostics, performing local edge filtering, and publishing telemetry over Wi-Fi via MQTT every 60 seconds.
