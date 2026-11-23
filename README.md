# Vehicle Predictive Maintenance Analytics

Integrated final project for the Master's Degree in Computer Science and Engineering (**Intelligent Embedded Systems** major) at **Alma Mater Studiorum – University of Bologna** (A.Y. 2025/2026, November 2026)[cite: 1]:

- **Machine Learning & Data Mining (Codice 95531)** — Predictive modeling, leakage-free evaluation, Isolation Forest outlier analysis, and WEKA comparison[cite: 1, 2, 6].
- **Distributed Systems (Codice 87474)** — Blockchain-backed VIN digital twins, smart contracts, and immutable service records (Hyperledger Fabric)[cite: 1, 7, 25].
- **Embedded Systems & IoT (Codice 77780)** — ESP32 telemetry acquisition via OBD-II, local edge pre-filtering, and MQTT reporting[cite: 1, 6, 25].

**Authors:** Danial Khayatian, Dias Katrenov, Mary Anne Selirio[cite: 1]  
**Supervisor:** Prof. Matteo Golfarelli[cite: 1]

---

## Repository Contents

- `Predictive_Maintenance.ipynb` — End-to-end Python pipeline (EDA, feature engineering, stratified split, Isolation Forest analysis, scikit-learn pipelines, cross-validation, test evaluation, learning curves, permutation importance, decision tree rule inspection, and cost-benefit analysis)[cite: 2, 13, 14, 28].
- `Predictive_Maintenance_Analytics.pdf` — Full project report[cite: 1, 28].
- `Datasets/` — Raw dataset (`vehicle_maintenance_data.csv`, 50,000 rows × 20 columns) and WEKA format (`data.arff`)[cite: 9, 28].
- `weka/` — WEKA experiment outputs (`Adaboost-decietionStrump.txt`, `functions-logistic.txt`, `trees-randomForest.txt`) and experiment reports[cite: 28].

---

## Key Findings & Results

### 1. Data Understanding & Deterministic Rule
- The raw dataset contains **50,000 records** and **20 columns** (19 features + binary target `Need_Maintenance`: `40,498` positive / `81.0%`, `9,502` negative / `19.0%`), with zero missing values or exact duplicate rows[cite: 9, 10].
- Exploratory analysis indicates a likely synthetic dataset[cite: 2, 10, 27]:
  - `18,096` records (`36.2%`) have `Mileage` greater than `Odometer_Reading`[cite: 10].
  - Grouping by the six primary driver attributes (`Battery_Status`, `Brake_Condition`, `Reported_Issues`, `Maintenance_History`, `Service_History`, `Accident_History`) yields **6,479 unique combinations** with **zero label conflicts**, revealing a deterministic rule behind `Need_Maintenance`[cite: 2, 11, 12].

### 2. Leakage-Free Methodology & Outlier Analysis
- Date columns (`Last_Service_Date`, `Warranty_Expiry_Date`) are deterministically converted into `Days_Since_Service` and `Days_To_Warranty_End` relative to `2024-03-01` before splitting[cite: 12].
- A **stratified 80/20 split** (`40,000` train / `10,000` test) is performed prior to data-driven preprocessing, with scaling and one-hot encoding fitted strictly inside `ColumnTransformer` pipelines on training folds only[cite: 12, 13].
- An **Isolation Forest** (`200` trees, `contamination=0.01`) fitted on training features flags `400` records (`1.0%`)[cite: 13, 14]. Inspection shows they represent rare-combination but valid vehicles rather than corrupted entries, and **stratified 5-fold cross-validation** confirms removing them yields no performance gain[cite: 14, 15]. All `40,000` training records are retained[cite: 15].

### 3. Test-Set Performance & Cost-Benefit Analysis (10,000 Vehicles)
Using illustrative error costs of **€500 per False Negative** (missed maintenance) and **€50 per False Positive** (unnecessary inspection)[cite: 14]:

| Model | Accuracy | ROC-AUC | TN / FP / FN / TP | Error Cost | Savings vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (`class_weight='balanced'`)[cite: 13] | `96.0%`[cite: 15] | `0.988`[cite: 15] | `1,893` / `7` / `393` / `7,707`[cite: 16] | **€196,850**[cite: 18, 19] | Baseline[cite: 19] |
| **Random Forest** (`100` trees)[cite: 13] | `100.0%`[cite: 15] | `1.000`[cite: 15] | `1,900` / `0` / `0` / `8,100`[cite: 16] | **€0**[cite: 18, 19] | **€196,850** (`€19.68` / vehicle)[cite: 18, 19] |

*(Note: Because the dataset appears synthetic with a deterministic label rule, 100% accuracy and illustrative savings should not be expected directly on noisy real-world fleet telemetry[cite: 18, 25, 27].)*

### 4. WEKA Comparison (10,000 Test Instances)
- **AdaBoost (Decision Stumps):** `96.96%` accuracy (`Kappa = 0.8996`, `MCC = 0.900`)[cite: 21, 22].
- **Logistic Regression:** `96.02%` accuracy (`Kappa = 0.8680`, `ROC Area = 0.993`)[cite: 21, 22].
- **Random Forest (`-depth 10`):** `86.78%` accuracy (`Kappa = 0.3901`), illustrating how tree depth constraints affect rule capture[cite: 21, 22].

### 5. Integrated Architecture (ML + Blockchain + IoT)
- **ML Service:** Flask `/predict` endpoint serving the Random Forest pipeline for maintenance probability/risk classification alongside Isolation Forest anomaly flagging[cite: 23, 25].
- **Blockchain Layer:** Hyperledger Fabric VIN-based digital twin and smart contracts (e.g., triggering maintenance workflows when predicted probability exceeds `80%`) for tamper-evident service history[cite: 7, 25].
- **IoT Layer:** ESP32 nodes reading OBD-II diagnostics, performing local edge filtering, and publishing telemetry over Wi-Fi via MQTT every 60 seconds[cite: 25].
