# Intelligent Resume Screening with Hiring Probability Feedback Using Machine Learning

# AI‑Powered Resume Checker — Notebook Edition

This repository contains a single Jupyter Notebook — **`1test-MscProject-Revised-FINAL.ipynb`** — for analyzing resume–job fit using classic NLP, Sentence‑BERT embeddings, and an XGBoost classifier. It generates actionable, section‑aware feedback (e.g., missing skills/tools, degree/experience gaps) and saves deployable artifacts for later reuse.

> **Tested environment:** Python **3.10** (notebook metadata shows 3.10.18). We recommend running via **Anaconda** with **Jupyter (Notebook or Lab)**.

---

## What this notebook does

- Cleans and normalizes resume + job description text (HTML/PDF/DOCX support).
- Extracts sections and requirements (skills/tools, education, experience).
- Builds TF‑IDF + SVM **resume category** model and saves it as `resume_category_model.joblib`.
- Computes **Sentence‑BERT** similarities using `all-MiniLM-L6-v2` for semantic matching.
- Trains an **XGBoost** classifier for match probability; calibrates and saves:
  - `xgb_model.joblib` (trained model)
  - `xgb_scaler.pkl` (feature scaler)
- Produces interpretable features and gap analysis for feedback.
- Reads/writes CSV inputs like `Resume.csv`, `training_data.csv` if present.

---

## Repository contents

```
.
├─ 1test-MscProject-Revised-FINAL.ipynb   # Main notebook (run this)
├─ (generated after running)
│  ├─ resume_category_model.joblib
│  ├─ xgb_model.joblib
│  └─ xgb_scaler.pkl
├─ (optional data files placed here)
│  ├─ Resume.csv
│  └─ training_data.csv
```

---

## Quick start (Anaconda + Jupyter)

1. **Install Anaconda** (if you don’t have it): <https://www.anaconda.com/download>
2. **Create and activate an environment (Python 3.10 recommended):**
   ```bash
   conda create -n resumeai python=3.10 -y
   conda activate resumeai
   ```
3. **Install required packages (via `pip` inside the environment):**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn nltk spacy spacy-transformers                pdfplumber python-docx beautifulsoup4 sentence-transformers torch xgboost shap tqdm joblib
   ```

   > If you have issues with `torch` on your machine, install the CPU build:
   > ```bash
   > pip install torch --index-url https://download.pytorch.org/whl/cpu
   > ```

4. **Download language models/corpora (one‑time):**
   ```bash
   python -m spacy download en_core_web_trf
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
   ```

5. **Launch Jupyter from Anaconda:**
   ```bash
   jupyter lab    # or: jupyter notebook
   ```
   - Open **`1test-MscProject-Revised-FINAL.ipynb`**.
   - **Kernel → Change Kernel** to your `resumeai` env if needed.
   - **Run → Run All Cells** (top to bottom).

> **Tip:** The first run will download the Sentence‑BERT model **`all-MiniLM-L6-v2`** (internet required once). It will be cached under your HuggingFace/transformers cache directory for future offline runs.

---

## Inputs & outputs

### Inputs
- **`Resume.csv`** and **`training_data.csv`** (if you’re training/evaluating with your own data). Place them in the **same folder** as the notebook or update the paths in the code.
- The notebook can also read individual **PDF/DOCX** resumes (e.g., `resume(def).pdf`) using `pdfplumber`/`python-docx` + BeautifulSoup for HTML cleanup.

### Outputs (artifacts)
- **`resume_category_model.joblib`** — TF‑IDF + LinearSVC + label encoder dictionary for resume category classification.
- **`xgb_model.joblib`** — Trained XGBoost model for final match probability.
- **`xgb_scaler.pkl`** — Fitted `MinMaxScaler` used to normalize features before XGBoost.
- Evaluation metrics/plots in‑notebook (matplotlib/seaborn).

> Keep **`xgb_model.joblib`** and **`xgb_scaler.pkl`** together when loading for inference in a separate script/notebook.

---

## How to run the full pipeline

1. **Prepare data (optional):**
   - Put `Resume.csv` and `training_data.csv` in the notebook folder (or edit the paths in the cells that call `pd.read_csv(...)`).

2. **Open the notebook** in Jupyter and run cells top→bottom:
   - **Imports + setup** (downloads NLTK resources if missing).
   - **Cleaning & feature extraction** (functions like `clean_text`, `extract_sections`, `extract_job_requirements`, etc.).
   - **Resume category model** (saves `resume_category_model.joblib`).
   - **Sentence‑BERT similarities** (loads `all-MiniLM-L6-v2`).
   - **Feature engineering → scaling → XGBoost training** (saves `xgb_model.joblib`, `xgb_scaler.pkl`).
   - **Evaluation + feedback generation** (functions like `generate_resume_feedback_cached`, tool/degree/experience checks).

3. **Predict on new pairs (resume + job description):**
   - Look for cells that read a resume file (PDF/DOCX) or a text string and a JD string/CSV.
   - Ensure the scaler/model are loaded:
     ```python
     import joblib
     model = joblib.load("xgb_model.joblib")
     scaler = joblib.load("xgb_scaler.pkl")
     ```
   - Use the provided helper functions to create features, scale them, and call `model.predict_proba(...)`.
   - The feedback helpers (e.g., missing tools/skills) can be used to generate section‑aware guidance.

---

## Troubleshooting

- **`ModuleNotFoundError` (spacy_transformers / en_core_web_trf)**  
  Ensure you ran both:
  ```bash
  pip install spacy spacy-transformers
  python -m spacy download en_core_web_trf
  ```

- **`LookupError` (NLTK stopwords/wordnet)**  
  Run:
  ```bash
  python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
  ```

- **Torch install issues**  
  Use the CPU build (above) or consult PyTorch’s official install selector.

- **Model files not found (`xgb_model.joblib`, `xgb_scaler.pkl`)**  
  Make sure you’ve run the training cells that call `joblib.dump(...)`. Keep the files in the same folder as the notebook when loading.

- **CSV not found**  
  Place `Resume.csv` / `training_data.csv` next to the notebook or update their paths in `pd.read_csv(...)` calls.

---

## Notes

- The notebook imports (non‑exhaustive): `numpy`, `pandas`, `matplotlib`, `seaborn`, `nltk`, `spacy`, `pdfplumber`, `python-docx`, `bs4` (`beautifulsoup4`), `sentence-transformers`, `torch`, `scikit-learn`, `xgboost`, `shap`, `tqdm`, `joblib`.
- Sentence‑BERT model: **`all-MiniLM-L6-v2`**.
- spaCy model: **`en_core_web_trf`** (transformer‑based pipeline).

---

## How to cite/acknowledge

- **Sentence‑BERT**: Reimers, N. & Gurevych, I. (2019). Sentence‑BERT: Sentence Embeddings using Siamese BERT‑Networks.  
- **spaCy** + **spaCy Transformers**.  
- **XGBoost**: Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.

---

## License

If you plan to publish this repository, choose and include a license (e.g., MIT). Otherwise, leave this section as a placeholder.
