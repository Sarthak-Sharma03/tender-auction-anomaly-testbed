# Tender & Auction Traffic Anomaly Detection Testbed

Open-source **testbed** for experimenting with **unsupervised anomaly detection** on tender / auction–style traffic (login and bidding events). The goal is to approximate how a security analytics system can flag **suspicious users, bids, or access patterns** by combining classical and deep anomaly-detection methods into a robust ensemble and exposing them through a simple web interface.

The implementation works on realistic event-log style data (user, IP/geo, device, browser, timestamps, success/failure flags) and is intended as a **research and prototyping sandbox**, not a production system. The same pipeline can be adapted to real tender / auction bidding logs or other high-risk transactional domains.

---

## Core Capabilities

- **Security-oriented event log modeling**
  - Designed for structured events such as logins and bids.
  - Extensible to tender / auction platforms with minimal schema changes.

- **Rich feature engineering**
  - Time-based features: hour-of-day, day-of-week, etc.
  - Frequency/encoding for high-cardinality categorical fields:
    - IP, country, region, city
    - Device type
    - Browser name and version
  - Scaling/normalization so distance-based and manifold methods behave sensibly.

- **Ensemble anomaly scoring**
  - Combines multiple heterogeneous detectors into a single anomaly score.
  - Percentile-based thresholds (e.g., top 5% most anomalous events) for flagging.
  - Inspired by contemporary practice on combining diverse detectors for more stable anomaly detection.

- **Deep representation learning**
  - Autoencoder to learn a compact representation of “normal” behavior.
  - Reconstruction error used as an additional anomaly signal.
  - Exploratory notebooks for deep anomaly-detection methods (e.g., DAGMM / Deep SVDD) alongside classical models.

- **End-to-end demo**
  - Notebooks for data exploration, model comparison, and ensemble design.
  - Saved artifacts (encoders, scalers, ensemble, autoencoder).
  - Flask web app + HTML UI to score new events interactively.

---

## Methods & Algorithms

During experimentation, several families of methods were benchmarked.

### Classical anomaly detectors

- IsolationForest  
- Local Outlier Factor (LOF)  
- One-Class SVM  
- Gaussian Mixture Models (GMM) with likelihood-based anomaly scores  

### Clustering-based methods

- K-Means clustering  
- DBSCAN and related density-based approaches  

These can be used both for exploratory structure discovery and as anomaly detectors (e.g., small or low-density clusters).

### Dimensionality reduction & visualization

- Principal Component Analysis (PCA)  
- t-SNE (t-distributed Stochastic Neighbor Embedding)  

Used primarily in the notebooks to understand feature-space structure and how different detectors behave relative to each other.

### Deep / representation-learning methods

- Autoencoder  
  - Trained to reconstruct “normal” events.  
  - Reconstruction error treated as an anomaly score.

- DAGMM / Deep SVDD (exploratory)  
  - Deep anomaly-detection architectures tested and compared against classical ensembles at a prototype level.

### Ensemble logic

Multiple detectors are combined into a single **ensemble anomaly score**:

- Scores from individual models are normalized and aggregated.  
- Percentile-based thresholds (e.g., top 5% most anomalous events) are used to mark suspicious events.  
- The ensemble design aims to reduce sensitivity to any single model’s failure modes and to capture different types of anomalies (global vs. local, density vs. reconstruction, etc.).

---

## Installation

1. Clone the repository:

    git clone https://github.com/your-username/tender-auction-anomaly-testbed.git
    cd tender-auction-anomaly-testbed

2. (Optional) Create and activate a virtual environment:

    python -m venv .venv
    source .venv/bin/activate   # On Windows: .venv\Scripts\activate

3. Install dependencies:

    pip install -r requirements.txt

---

## Usage

### 1. Run the web demo

Start the Flask app:

    python app.py

This starts a Flask server (typically on `http://127.0.0.1:5000/`).

Open the URL in your browser to access the UI:

1. Fill in one or more events (e.g., login or bidding attempts) with:
   - User ID  
   - Timestamp  
   - IP and location fields (country, region, city)  
   - Device type  
   - Browser name and version  
   - Success/failure flag or similar indicator  

2. Submit the form.

3. The backend:
   - Applies the same preprocessing pipeline used during training.  
   - Runs the ensemble of detectors (classical + autoencoder).  
   - Returns an anomaly score and a normal/anomalous decision for each event.

### 2. Experiment and retrain (optional)

Use the notebooks in `Jupyter Notebook/` to:

- Explore the dataset: distributions, correlations, failure patterns.  
- Engineer and refine features (time, geo/IP, device/browser).  
- Benchmark and compare detectors:
  - IsolationForest, LOF, One-Class SVM, GMM  
  - K-Means / DBSCAN  
  - Autoencoder-based reconstruction  
- Build and tune the ensemble and thresholds.  
- Save updated model artifacts (encoders, scalers, ensemble, autoencoder) for the Flask app.

To plug in your own tender/auction bidding logs or other event logs:

- Adapt the feature-engineering steps to your schema.  
- Regenerate model artifacts via the notebooks.  

---

## Dataset

The repository includes a **sample login-style dataset**, for example:

- `data/Login_Data.xlsx`

It mimics event logs with fields such as:

- User identifiers  
- IP and coarse geo information (country, region, city)  
- Device type and browser (name + version)  
- Timestamp and success/failure flags  

This dataset is intended as a **proxy** for tender / auction traffic:

- It exhibits realistic patterns such as repeated logins, failures, typical time-of-day usage.  
- It is sufficient to demonstrate and stress-test the anomaly-detection pipeline.

For real-world use, you should:

- Replace it with your own domain-specific event logs (e.g., bidding histories, tender submissions).  
- Revisit feature engineering to reflect domain knowledge (e.g., bid increments, tender deadlines, time-to-deadline behavior).  
- Carefully tune thresholds and evaluation metrics.  

---

## Project Structure

Approximate layout (adapt path names if your repo differs):

- `app.py`  
  Flask web server exposing the anomaly-detection demo UI.

- `data.py` (if present)  
  Utilities related to loading / handling the sample dataset.

- `data/Login_Data.xlsx`  
  Sample event dataset used for experiments.

- `Jupyter Notebook/`  
  - `(1)Data_Experimentation.ipynb` – Data exploration and feature engineering.  
  - `(2)Analysis_and_Model_Selection.ipynb` – Model comparison and selection.  
  - `(3)Models_and_Ensemble.ipynb` – Building, evaluating, and exporting the ensemble.

- Model & preprocessing artifacts (names may vary):  
  - `encoder.joblib`  
  - `scaler.pkl`  
  - `AutoE.pkl`  
  - `ensemble.pkl`  
  - `pipe.pkl`  
  - `pre_pro1.pkl`, `pre_pro2.pkl`  

- `templates/index.html`  
  HTML template for the web UI.

- `screenshots/`  
  Sample screenshots of the UI and results.

- `Video/Demo.mp4`  
  Short demo video of the system in action.

- `UsageReport.pdf`  
  High-level summary/report of the project and experimental findings.

- `requirements.txt`  
  Python dependencies.

---

## Background & Motivation

Modern tender and auction platforms handle high volumes of sensitive transactional traffic and are exposed to risks such as:

- Automated or scripted bidding behavior  
- Account takeover and credential stuffing  
- Abnormal access patterns tied to collusion or abuse  

This testbed explores how a combination of **unsupervised anomaly detectors** and **deep representation learning** can be used to:

- Model typical user and bidding behavior.  
- Surface outlying events for further investigation.  
- Provide a flexible playground for tuning thresholds and strategies before any production deployment.  

---

## Disclaimer

This repository is intended as a **research and educational testbed** for security-focused anomaly detection on event logs (e.g., tender / auction or login traffic). It is **not** a hardened, production-ready security product.

Any real-world deployment should include, at minimum:

- Domain-specific tuning and calibration.  
- Rigorous offline and online evaluation against real attack data and false-positive costs.  
- Monitoring, alerting, and human-in-the-loop review.  
- Appropriate legal, compliance, security, and privacy considerations.  
