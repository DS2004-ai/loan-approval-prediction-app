# =========================
# Core Libraries & Imports
# =========================
# Standard library
import os          # For file path handling (locating the CSV relative to this script)
import warnings    # To silence non-critical warnings and keep UI tidy

# Data science stack
import numpy as np             # Numerical operations, arrays
import pandas as pd            # DataFrames (tabular data)

# Visualization
import plotly.express as px    # Quick interactive charts (histograms, bars, scatter, box)
import plotly.graph_objects as go  # Custom Plotly figures (e.g., gauge)
import matplotlib.pyplot as plt     # Matplotlib for heatmaps, ROC, etc.
import seaborn as sns               # Nicer heatmaps built on matplotlib

# Preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # For categorical encoding + numeric scaling

# Modeling & Evaluation
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict  # CV and out-of-fold predictions
from sklearn.naive_bayes import GaussianNB          # Naive Bayes classifier (Gaussian variant)
from sklearn.inspection import permutation_importance  # Model-agnostic feature importance
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Decision Tree classifier and visualization
from sklearn.metrics import (   # Suite of classification metrics
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, roc_auc_score
)
from sklearn.base import clone  # Safe cloning/retraining of sklearn estimators when features change

# App framework
import streamlit as st  # Streamlit for UI

# Keep the console/UI clean of non-critical warnings
warnings.filterwarnings('ignore')


# =========================
# Paths & Data Loading
# =========================
# We resolve the CSV path relative to THIS script so the app works no matter the working directory.
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "Loan approval.csv")

# Load the original dataset (untouched). We'll keep this for EDA displays.
RawData = pd.read_csv(file_path)


# =========================
# Preprocessing (Label Enc., Outlier Capping, Features)
# =========================
# Work on a COPY so we never mutate RawData.
LoanData = RawData.copy()

# 1) Remove non-predictive ID if present.
if "Loan_ID" in LoanData.columns:
    LoanData.drop("Loan_ID", axis=1, inplace=True)

# 2) Strictly drop rows with any missing values (simple, consistent policy for this demo).
LoanData.dropna(inplace=True)

# 3) Create numerical target column: Loan_Status "Y"/"N" -> 1/0, then drop the original text column.
LoanData["Target"] = LoanData["Loan_Status"].map({"Y": 1, "N": 0})
LoanData.drop("Loan_Status", axis=1, inplace=True)

# 4) Label-encode binary categoricals into 0/1 for modeling (DT likes numeric).
label_encoders = {}
binary_cols = ["Gender", "Married", "Education", "Self_Employed"]
for col in binary_cols:
    if col in LoanData.columns:
        le = LabelEncoder()
        LoanData[col] = le.fit_transform(LoanData[col])
        label_encoders[col] = le  # Store encoder (useful if you later persist and reload)

# 5) Dependents sometimes has a "3+" category. Normalize that into an integer 3.
if "Dependents" in LoanData.columns:
    # Strip the plus sign literally (not regex-based).
    LoanData["Dependents"] = LoanData["Dependents"].astype(str).str.replace("+", "", regex=False)
    # Coerce to numeric; unknowns become NaN -> fill with 0 -> cast to int.
    LoanData["Dependents"] = pd.to_numeric(LoanData["Dependents"], errors="coerce").fillna(0).astype(int)

# 6) Property_Area may be text. Encode it as integers (e.g., Rural=0, Semiurban=1, Urban=2).
if "Property_Area" in LoanData.columns and LoanData["Property_Area"].dtype == "object":
    le = LabelEncoder()
    LoanData["Property_Area"] = le.fit_transform(LoanData["Property_Area"])

# 7) Cap outliers on skewed numeric variables using IQR rule to reduce undue influence.
def iqr_cap_series(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return series.clip(lower, upper)

for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
    if col in LoanData.columns:
        LoanData[col] = iqr_cap_series(LoanData[col], factor=1.5)

# 8) Rename a few columns to clearer boolean-style names for readability in UI and plots.
LoanData.rename(
    columns={
        "Gender": "IsMale",
        "Married": "IsMarried",
        "Education": "IsGraduate",
        "Self_Employed": "IsSelfEmployed",
        "Credit_History": "HasGoodCredit"
    },
    inplace=True
)

# 9) FEATURE ENGINEERING (important!)
#    - TotalIncome = ApplicantIncome + CoapplicantIncome
#    - EMI = LoanAmount / Loan_Amount_Term  (we’re not using interest here; treat as simple monthly obligation)
#    - DTI = EMI / TotalIncome  (Debt-to-Income ratio; higher means more burden relative to income)
LoanData["TotalIncome"] = LoanData["ApplicantIncome"] + LoanData["CoapplicantIncome"]

# Median-impute zeros/missing Loan_Amount_Term for robust EMI calculation
term_safe = LoanData["Loan_Amount_Term"].replace(0, np.nan)
term_safe = term_safe.fillna(term_safe.median())
emi = (LoanData["LoanAmount"] / term_safe).replace([np.inf, -np.inf], np.nan).fillna(0)
LoanData["EMI"] = emi

# DTI handling: protect against division by zero by replacing zeros with median TotalIncome
ti_safe = LoanData["TotalIncome"].replace(0, np.nan).fillna(LoanData["TotalIncome"].median())
LoanData["DTI"] = (LoanData["EMI"] / ti_safe).replace([np.inf, -np.inf], np.nan).fillna(0)

# Human-friendly mapping for UI dropdowns
AREA_MAP = {0: "Rural", 1: "Semiurban", 2: "Urban"}

# 10) Keep Target as the LAST column (nice for viewing tables).
if "Target" in LoanData.columns:
    cols = [c for c in LoanData.columns if c != "Target"] + ["Target"]
    LoanData = LoanData[cols]


# =========================
# Scaled / One-Hot Encoded Copy (for NB)
# =========================
# Why a second dataset? Naive Bayes (Gaussian) behaves better when
# features are scaled and categoricals are expanded into 0/1 dummy columns.
cols_to_scale = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome', 'EMI', 'DTI']

# Scale numeric columns into [0,1] range
scaler = MinMaxScaler(feature_range=(0, 1))
LoanData_scaled = LoanData.copy()
LoanData_scaled.loc[:, cols_to_scale] = scaler.fit_transform(LoanData[cols_to_scale])

# One-hot encode Property_Area and Dependents into dummy columns (keep all levels).
ohe_cols = []
if "Property_Area" in LoanData_scaled.columns:
    ohe_cols.append("Property_Area")
if "Dependents" in LoanData_scaled.columns:
    ohe_cols.append("Dependents")

if ohe_cols:
    already_ohe = any(c.startswith("PA_") or c.startswith("Dep_") for c in LoanData_scaled.columns)
    if not already_ohe:
        LoanData_scaled = pd.get_dummies(
            LoanData_scaled,
            columns=ohe_cols,
            prefix=["PA" if c == "Property_Area" else "Dep" for c in ohe_cols],
            drop_first=False  # Keep ALL levels to avoid implied baselines
        )

# Make sure one-hot columns are ints (0/1), not booleans
dummy_cols = [c for c in LoanData_scaled.columns if c.startswith("PA_") or c.startswith("Dep_")]
if dummy_cols:
    LoanData_scaled[dummy_cols] = LoanData_scaled[dummy_cols].astype(int)

# Keep Target last here too
if "Target" in LoanData_scaled.columns:
    cols = [c for c in LoanData_scaled.columns if c != "Target"] + ["Target"]
    LoanData_scaled = LoanData_scaled[cols]


# =========================
# Decision Tree Defaults & Baseline Fit
# =========================
# We use TotalIncome instead of CoapplicantIncome and include EMI (derived burden proxy).
dt_features_default = ["HasGoodCredit", "TotalIncome", "EMI", "LoanAmount", "Property_Area", "IsMarried"]
target_col = "Target"

# Train a small, interpretable baseline tree (max_depth=3) for the Prediction page fallback.
X_dt_global = LoanData[dt_features_default]
y_dt_global = LoanData[target_col]
dt_default_threshold = 0.50  # Default decision threshold for DT page (can be tuned in UI)
nb_default_threshold = 0.50  # Default decision threshold for NB page (tuned per CV)

model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=1,
    min_samples_split=2,
    criterion="gini",
    random_state=42
)
model.fit(X_dt_global, y_dt_global)

# Expose a working model & feature list into session on first load.
if "dt_model" not in st.session_state:
    st.session_state["dt_model"] = model
    st.session_state["dt_features"] = dt_features_default


# =========================
# Utility: Guard against feature mismatches
# =========================
def ensure_dt_model(mdl, X_all, y_all):
    """
    If a model in session was fitted on OLD feature names/order, refit it on the CURRENT X.
    Prevents the common "feature names mismatch" error when UI-driven features change.
    """
    try:
        expected = list(getattr(mdl, "feature_names_in_", []))
        current = list(X_all.columns)
        # If the saved model was trained with different feature columns/order -> refit clone on current data.
        if expected and expected != current:
            mdl = clone(mdl).fit(X_all, y_all)
    except Exception:
        # Fail silently: worst case, the normal training path will run later.
        pass
    return mdl


# =========================
# Streamlit Page: Dataset Views
# =========================
def page1():
    """Basic toggles to view the raw, processed, and scaled datasets."""
    st.subheader("📄 Loan Approval Datasets")
    if st.checkbox("Raw Loan Approval Data", key="chk_raw"):
        st.write(RawData)
    if st.checkbox("IQR + Label Encoding + Feature Engineering (LoanData)", key="chk_proc"):
        st.write(LoanData)
    if st.checkbox("Scaled + One-Hot Encoded (LoanData_scaled)", key="chk_scaled"):
        st.write(LoanData_scaled)


# =========================
# Streamlit Page: EDA (Plotly on RawData)
# =========================
def page2():
    """
    Lightweight EDA on the original dataset (RawData).
    Purpose: understand distributions, relationships, and label balance BEFORE modeling.
    """
    st.subheader("📊 Exploratory Data Analysis (Plotly, RawData)")

    raw_df = RawData.copy()

    # Infer numeric vs categorical columns
    num_cols = raw_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = raw_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # KPI tiles (quick facts)
    c1, c2, c3, c4 = st.columns(4)
    total_rows = len(raw_df)
    approved = raw_df["Loan_Status"].eq("Y").sum() if "Loan_Status" in raw_df.columns else None
    not_approved = raw_df["Loan_Status"].eq("N").sum() if "Loan_Status" in raw_df.columns else None
    med_income = raw_df["ApplicantIncome"].median() if "ApplicantIncome" in raw_df.columns else None

    with c1: st.metric("Total Records", f"{total_rows:,}")
    with c2: st.metric("Approved (Y)", f"{approved:,}" if approved is not None else "—")
    with c3: st.metric("Not Approved (N)", f"{not_approved:,}" if not_approved is not None else "—")
    with c4: st.metric("Median Applicant Income", f"{med_income:,.0f}" if med_income is not None else "—")

    st.divider()

    # Histograms for numeric columns (optionally colored by Loan_Status if present)
    st.markdown("### 📈 Histograms (Numeric)")
    if num_cols:
        choose_nums = st.multiselect("Choose numeric columns", num_cols, default=num_cols[:6], key="eda_num_cols")
        facet_target = st.checkbox("Facet / color by Loan_Status (if present)?", value=True, key="eda_facet")
        for col in choose_nums:
            if facet_target and "Loan_Status" in raw_df.columns:
                fig = px.histogram(raw_df, x=col, nbins=40, color="Loan_Status",
                                   marginal="box", title=f"Distribution of {col} by Loan_Status")
            else:
                fig = px.histogram(raw_df, x=col, nbins=40, marginal="box", title=f"Distribution of {col}")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numerical columns found.")

    st.divider()

    # Correlation heatmap among numeric variables (helps spot collinearity)
    st.markdown("### 🔗 Correlation Heatmap (Numerical Features)")
    if len(num_cols) >= 2:
        corr = raw_df[num_cols].corr().round(2)
        heat = go.Figure(
            data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale="Blues", zmin=-1, zmax=1,
                colorbar=dict(title="corr")
            )
        )
        heat.update_layout(height=520, title="Correlation Heatmap")
        st.plotly_chart(heat, use_container_width=True)
    else:
        st.info("Need at least two numerical columns to show a correlation matrix.")

    st.divider()

    # A few additional useful splits/plots
    st.markdown("### 🧭 Additional Visuals")

    if "Loan_Status" in raw_df.columns:
        loan_status_counts = (
            raw_df["Loan_Status"].value_counts(dropna=False).rename_axis("Loan_Status").reset_index(name="Count")
        )
        fig = px.bar(loan_status_counts, x="Loan_Status", y="Count", text_auto=True, title="Loan Status Counts")
        st.plotly_chart(fig, use_container_width=True)

    if {"Property_Area", "Loan_Status"}.issubset(raw_df.columns):
        fig = px.histogram(raw_df, x="Property_Area", color="Loan_Status",
                           barmode="group", text_auto=True, title="Loan Status by Property Area")
        st.plotly_chart(fig, use_container_width=True)

    if {"ApplicantIncome", "LoanAmount"}.issubset(raw_df.columns):
        color_col = "Loan_Status" if "Loan_Status" in raw_df.columns else None
        fig = px.scatter(raw_df, x="ApplicantIncome", y="LoanAmount",
                         color=color_col, hover_data=raw_df.columns,
                         title="Applicant Income vs Loan Amount")
        st.plotly_chart(fig, use_container_width=True)

    if {"LoanAmount", "Education"}.issubset(raw_df.columns):
        fig = px.box(raw_df, x="Education", y="LoanAmount", color="Education",
                     title="Loan Amount by Education")
        st.plotly_chart(fig, use_container_width=True)

    if {"LoanAmount", "Property_Area"}.issubset(raw_df.columns):
        fig = px.box(raw_df, x="Property_Area", y="LoanAmount", color="Property_Area",
                     title="Loan Amount by Property Area")
        st.plotly_chart(fig, use_container_width=True)

    if {"Credit_History", "Loan_Status"}.issubset(raw_df.columns):
        fig = px.histogram(raw_df, x="Credit_History", color="Loan_Status",
                           barmode="group", text_auto=True, title="Loan Status by Credit History")
        st.plotly_chart(fig, use_container_width=True)


# =========================
# Streamlit Page: Decision Tree (CV + Manual / Grid optional)
# =========================
def page3():
    """
    Decision Tree classifier evaluated via 10-fold Stratified CV.
    - GridSearchCV is OPTIONAL and OFF by default (you can toggle it on).
    - Otherwise, tune simple parameters with sliders.
    - Decision threshold (probability cutoff) is adjustable for business trade-offs.
    """
    st.subheader("🌳 Decision Tree — 10-Fold Cross-Validation")

    # Use the DT default feature set that includes TotalIncome and EMI (as requested).
    features = dt_features_default
    target = target_col
    X = LoanData[features]
    y = LoanData[target]

    # ---- Sidebar controls
    st.sidebar.header("🛠️ Decision Tree Options")
    # Keep GridSearch optional and OFF by default.
    use_grid = st.sidebar.checkbox("Use GridSearchCV", value=False, key="dt_use_grid")
    refit_metric = st.sidebar.selectbox(
        "Optimize (refit) for",
        options=["f1", "precision", "recall", "accuracy"],
        index=0,
        key="dt_refit"
    )

    # Business threshold for approval decision (post-proba cutoff)
    st.sidebar.markdown("### 🎚️ Decision Threshold (Tree)")
    dt_threshold = st.sidebar.slider(
        "Approve if P(Approved) ≥",
        0.10, 0.95, 0.50, 0.01,
        key="dt_threshold_slider"
    )
    st.session_state["dt_threshold"] = float(dt_threshold)

    # 10-fold stratified CV keeps target ratio similar across folds.
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # ---- Option A: GridSearchCV (if toggled on)
    if use_grid:
        param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [2, 3, 4, 5, 6],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10],
            "ccp_alpha": [0.0, 0.001, 0.002, 0.005],
            "class_weight": [None, {0: 1, 1: 1}, {0: 2, 1: 1}],
            "splitter": ["best"]
        }
        # Score on multiple metrics; refit (choose best) according to the dropdown.
        scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}

        grid = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid=param_grid,
            scoring=scoring,
            refit=refit_metric,
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X, y)
        best_model = grid.best_estimator_

        st.write("#### ✅ Best hyperparameters:")
        st.write(grid.best_params_)
        st.write(f"Best mean CV {refit_metric}: {grid.best_score_:.4f}")

    # ---- Option B: Manual sliders (DEFAULT)
    else:
        max_depth = st.sidebar.slider("Max Depth", 1, 20, 3, key="dt_max_depth")
        min_samples_leaf = st.sidebar.slider("Min Samples per Leaf", 1, 100, 10, key="dt_min_leaf")
        min_samples_split = st.sidebar.slider("Min Samples to Split", 2, 200, 10, key="dt_min_split")
        criterion = st.sidebar.selectbox("Split Criterion", ["gini", "entropy"], key="dt_criterion")
        class_weight_opt = st.sidebar.selectbox("Class Weight", ["None", "Balanced"], index=0, key="dt_cw")
        class_weight = None if class_weight_opt == "None" else "balanced"

        best_model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            criterion=criterion,
            class_weight=class_weight,
            random_state=42
        )

    # ---- Cross-validated OUT-OF-FOLD probabilities -> realistic generalization metrics
    prob_cv = cross_val_predict(best_model, X, y, cv=cv, method="predict_proba")[:, 1]
    # Turn probabilities into 0/1 based on business threshold
    y_pred_cv = (prob_cv >= dt_threshold).astype(int)

    # Metrics (OOF)
    acc = accuracy_score(y, y_pred_cv)
    prec = precision_score(y, y_pred_cv, zero_division=0)
    rec = recall_score(y, y_pred_cv, zero_division=0)
    f1 = f1_score(y, y_pred_cv, zero_division=0)
    auc = roc_auc_score(y, prob_cv)

    st.markdown("### 📏 Cross-Validated Metrics (Out-of-fold)")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**Precision:** {prec:.4f}")
    st.write(f"**Recall:** {rec:.4f}")
    st.write(f"**F1-score:** {f1:.4f}")
    st.write(f"**ROC AUC:** {auc:.4f}")

    # Confusion matrix (actual vs predicted classes)
    cm = confusion_matrix(y, y_pred_cv)
    st.markdown("### 🧮 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
        xticklabels=["Not Approved", "Approved"],
        yticklabels=["Not Approved", "Approved"]
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # ROC curve (TPR vs FPR across all thresholds)
    fpr, tpr, _ = roc_curve(y, prob_cv)
    st.markdown("### 📈 ROC Curve (10-Fold CV Probabilities)")
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # Fit the chosen model on ALL data for later prediction usage.
    best_model.fit(X, y)
    st.session_state["dt_model"] = best_model
    st.session_state["selected_features"] = features
    st.session_state["dt_features"] = features
    st.success("✅ Tree updated — available on the Prediction & Interpretation pages.")

    # Visualize the final trained tree
    st.markdown("### 🌳 Tree Structure (trained on all data)")
    fig_tree, ax_tree = plt.subplots(figsize=(12, 6))
    plot_tree(
        best_model,
        feature_names=features,
        class_names=["Not Approved", "Approved"],
        filled=True, rounded=True, fontsize=10, ax=ax_tree
    )
    st.pyplot(fig_tree)


# =========================
# Streamlit Page: Naive Bayes (GaussianNB, simple UI; no var_smoothing knobs)
# =========================
def page4():
    """
    Gaussian Naive Bayes classifier evaluated via 10-fold Stratified CV.
    - No GridSearch UI and no var_smoothing knobs (as requested).
    - We still optimize the decision threshold (probability cutoff) for a chosen metric.
    """
    st.subheader("🧠 Naive Bayes — 10-Fold CV (GaussianNB)")

    # Guard: ensure the scaled/OHE dataset exists
    if "LoanData_scaled" not in globals():
        st.error("`LoanData_scaled` not found. Please create it before using this page.")
        st.stop()
    df = LoanData_scaled

    # Base NB features: scaled numerics + one-hot dummies + a few key binaries
    base_feats = ["HasGoodCredit", "CoapplicantIncome", "LoanAmount", "IsMarried", "TotalIncome", "EMI", "DTI"]
    pa_cols  = [c for c in df.columns if c.startswith("PA_")]
    dep_cols = [c for c in df.columns if c.startswith("Dep_")]
    features = base_feats + pa_cols + dep_cols
    target = "Target"

    # Guard: confirm all features exist
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        st.error(f"Missing columns in LoanData_scaled: {missing}")
        st.stop()

    X = df[features].copy()
    y = df[target]
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # UI: choose which metric to optimize the threshold for
    st.sidebar.header("🛠️ Naive Bayes Options")
    optimize_for = st.sidebar.selectbox(
        "Optimize threshold for", ["accuracy", "f1", "precision", "recall"], index=0, key="nb_opt_metric"
    )

    # Plain GaussianNB (no hyperparameter UI; keep it simple/clean)
    nb_model = GaussianNB()

    # Cross-validated OOF probabilities
    prob_cv = cross_val_predict(nb_model, X, y, cv=cv, method="predict_proba")[:, 1]

    # Helper: evaluate a metric at a specific threshold
    def metric_at_threshold(y_true, p1, thr, metric):
        yp = (p1 >= thr).astype(int)
        if metric == "accuracy":
            return accuracy_score(y_true, yp)
        elif metric == "f1":
            return f1_score(y_true, yp, zero_division=0)
        elif metric == "precision":
            return precision_score(y_true, yp, zero_division=0)
        elif metric == "recall":
            return recall_score(y_true, yp, zero_division=0)
        return accuracy_score(y_true, yp)  # default

    # Search over a grid of thresholds and keep the best for the chosen metric
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = [metric_at_threshold(y, prob_cv, t, optimize_for) for t in thresholds]
    best_idx = int(np.argmax(scores))
    best_t = float(thresholds[best_idx])
    st.session_state["nb_threshold"] = float(best_t)

    # Report CV metrics at the chosen threshold
    y_pred_cv = (prob_cv >= best_t).astype(int)
    acc = accuracy_score(y, y_pred_cv)
    prec = precision_score(y, y_pred_cv, zero_division=0)
    rec = recall_score(y, y_pred_cv, zero_division=0)
    f1  = f1_score(y, y_pred_cv, zero_division=0)
    auc = roc_auc_score(y, prob_cv)

    st.markdown("### 📏 Cross-Validated Metrics (10-Fold, Out-of-Fold)")
    st.write(f"**Optimized for:** `{optimize_for}`  |  **Best threshold:** `{best_t:.2f}`")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**Precision:** {prec:.4f}")
    st.write(f"**Recall:** {rec:.4f}")
    st.write(f"**F1-score:** {f1:.4f}")
    st.write(f"**ROC AUC:** {auc:.4f}")
    st.caption(f"Features ({len(features)}): {features[:8]}{' ...' if len(features)>8 else ''}")

    # Confusion matrix visualization
    cm = confusion_matrix(y, y_pred_cv)
    st.markdown("### 🧮 Confusion Matrix (10-Fold CV Predictions)")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
        xticklabels=["Not Approved", "Approved"],
        yticklabels=["Not Approved", "Approved"]
    )
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # ROC curve visualization
    fpr, tpr, _ = roc_curve(y, prob_cv)
    st.markdown("### 📈 ROC Curve (10-Fold CV Probabilities)")
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # Fit NB on ALL data (so Prediction page can use it immediately)
    nb_model.fit(X, y)
    st.session_state["nb_model"] = nb_model
    st.session_state["nb_features"] = features
    st.session_state["nb_pa_cols"] = pa_cols
    st.session_state["nb_dep_cols"] = dep_cols

    st.success("✅ Naive Bayes updated — available on the Prediction page.")


# =========================
# Streamlit Page: Prediction (Quick what-if)
# =========================
def page5():
    """
    A friendly 'what-if' page:
    - Decision Tree prediction uses RAW inputs; we silently compute EMI from user LoanAmount & Loan_Amount_Term.
    - Naive Bayes prediction uses the scaled/OHE feature space but the UI stays simple; we transform under the hood.
    """
    st.subheader("🔮 Quick Prediction")

    # Pull stored thresholds (or use defaults if a modeling page hasn't been visited yet)
    th_dt = float(st.session_state.get("dt_threshold", dt_default_threshold))
    th_nb = float(st.session_state.get("nb_threshold", nb_default_threshold))

    st.info(
        f"**Current thresholds** — Decision Tree: `{th_dt:.2f}`  |  Naive Bayes: `{th_nb:.2f}`\n\n"
        f"_Tip: tune DT on the Decision Tree page and NB on the Naive Bayes page._"
    )

    model_choice = st.selectbox(
        "Choose model", ["Decision Tree", "Naive Bayes"], index=0, key="home_model_choice"
    )

    # ---- Decision Tree predictor on raw inputs ----
    if model_choice == "Decision Tree":
        st.markdown("#### 🌳 Decision Tree (raw inputs)")
        dt_features = dt_features_default

        # Guard for missing columns
        missing_dt = [c for c in dt_features + ["Target"] if c not in LoanData.columns]
        if missing_dt:
            st.error(f"Missing columns in LoanData for Decision Tree: {missing_dt}")
            st.stop()

        # Ensure the DT model in session matches the current feature set
        X_all_dt = LoanData[dt_features]
        y_all_dt = LoanData[target_col]
        dt_model = st.session_state.get("dt_model", model)
        dt_model = ensure_dt_model(dt_model, X_all_dt, y_all_dt)
        st.session_state["dt_model"] = dt_model

        # UI inputs
        cols = st.columns(2)
        med = LoanData[dt_features].median(numeric_only=True)
        # We'll need a reasonable default for Loan_Amount_Term to compute EMI if user doesn't change it.
        term_med = float(LoanData["Loan_Amount_Term"].replace(0, np.nan).dropna().median()) \
            if "Loan_Amount_Term" in LoanData.columns else 360.0  # 360 months ~ 30-year mortgage

        with cols[0]:
            has_good_credit = st.selectbox("Has Good Credit? (1=Yes, 0=No)", [1, 0], index=0, key="dt_good_credit")
            is_married = st.selectbox("Is Married? (1=Yes, 0=No)", [1, 0], index=0, key="dt_is_married")
            pa_label_ui = st.selectbox("Property Area", list(AREA_MAP.values()), index=1, key="dt_property_area")
            property_area = [k for k, v in AREA_MAP.items() if v == pa_label_ui][0]

        with cols[1]:
            total_income = st.number_input(
                "Total Income (raw)",
                value=float(med.get("TotalIncome", 0.0)),
                step=100.0, key="dt_total_income"
            )
            loan_amount = st.number_input(
                "Loan Amount (raw)",
                value=float(med.get("LoanAmount", 100.0)),
                step=1.0, key="dt_loan_amount"
            )
            loan_term = st.number_input(
                "Loan Amount Term (months)",
                value=term_med,
                min_value=1.0,  # avoid division by zero
                step=1.0, key="dt_loan_term"
            )

        # Silently compute EMI from user inputs (no need to display it).
        emi_val = float(loan_amount) / float(loan_term) if float(loan_term) > 0 else 0.0

        # Build model input row (must match the feature list used during fitting).
        X_pred = pd.DataFrame(
            [[has_good_credit, total_income, emi_val, loan_amount, property_area, is_married]],
            columns=dt_features
        )

        # Predict
        if st.button("Predict with Decision Tree", key="btn_dt_predict"):
            prob = dt_model.predict_proba(X_pred)[0, 1]
            pred = int(prob >= th_dt)
            label = "Approved" if pred == 1 else "Not Approved"

            st.success(f"**Prediction:** {label}  |  **P(Approved)** = {prob:.2f}  |  **Threshold** = {th_dt:.2f}")

            # Friendly gauge to visualize probability vs threshold
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': '%'},
                title={'text': "Probability of Approval"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'threshold': {'line': {'width': 3}, 'thickness': 0.75, 'value': th_dt * 100}
                }
            ))
            st.plotly_chart(gauge, use_container_width=True)

    # ---- Naive Bayes predictor on scaled/OHE inputs ----
    else:
        st.markdown("#### 🧠 Naive Bayes (raw inputs → auto-scaled)")

        # Guard: model must be trained on NB page first
        if "nb_model" not in st.session_state:
            st.warning("Naive Bayes model not trained yet — open 'Classification: Naive Bayes' page first.")
            st.stop()

        nb_model    = st.session_state["nb_model"]
        nb_features = st.session_state["nb_features"]
        pa_cols     = st.session_state["nb_pa_cols"]
        dep_cols    = st.session_state["nb_dep_cols"]

        # Helpers to convert OHE column names back to human labels for UI
        def _pa_human_labels(pa_cols):
            if not pa_cols:
                return list(AREA_MAP.values())
            suffixes = [c.split("PA_", 1)[1] for c in pa_cols]
            labels = []
            for s in sorted(suffixes, key=lambda x: str(x)):
                try:
                    labels.append(AREA_MAP[int(s)])
                except Exception:
                    labels.append(s)
            return labels

        def _dep_values(dep_cols):
            # Dependents present in the model (e.g., Dep_0, Dep_1, Dep_2, Dep_3)
            if not dep_cols:
                return [0, 1, 2, 3]
            return sorted(int(c.split("Dep_", 1)[1]) for c in dep_cols)

        pa_labels_from_cols = _pa_human_labels(pa_cols)
        dep_values = _dep_values(dep_cols)

        # Display "3+" instead of 3 for better UX
        dep_display_labels = [("3+" if v >= 3 else str(v)) for v in dep_values]
        dep_choice = st.selectbox("Number of Dependents", dep_display_labels, index=0, key="nb_dependents_home")

        cols = st.columns(2)
        with cols[0]:
            has_good_credit = st.selectbox("Has Good Credit? (1=Yes, 0=No)", [1, 0], index=0, key="nb_good_credit_home")
            is_married      = st.selectbox("Is Married? (1=Yes, 0=No)", [1, 0], index=0, key="nb_is_married_home")
            pa_label        = st.selectbox("Property Area", pa_labels_from_cols, index=1, key="nb_property_area_home")

        with cols[1]:
            coapp_raw  = st.number_input("Coapplicant Income (raw)", min_value=0.0, value=0.0, step=100.0,
                                         key="nb_coapp_income_home")
            loan_raw   = st.number_input("Loan Amount (raw)",       min_value=0.0, value=0.0, step=100.0,
                                         key="nb_loan_amount_home")

        # Scale & engineer the exact numeric features NB was trained on
        # We mimic training-time transforms: compute TotalIncome, EMI (with dataset median term), DTI; then MinMax scale.
        def scale_and_engineer(coapp_raw, loan_raw):
            appl_med = float(LoanData["ApplicantIncome"].median())
            term_med = float(LoanData["Loan_Amount_Term"].replace(0, np.nan).dropna().median())
            total_income = appl_med + coapp_raw
            emi = 0.0 if term_med == 0 else loan_raw / term_med
            dti = 0.0 if total_income == 0 else emi / total_income
            row = pd.DataFrame([[appl_med, coapp_raw, loan_raw, total_income, emi, dti]],
                               columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                        'TotalIncome', 'EMI', 'DTI'])
            scaled = scaler.transform(row)  # Use the same scaler fitted earlier
            return dict(zip(row.columns, scaled.ravel()))

        svals = scale_and_engineer(coapp_raw, loan_raw)

        # Build the model input row aligned with the NB training feature set
        input_row = pd.DataFrame(np.zeros((1, len(nb_features))), columns=nb_features)
        input_row.loc[:, "HasGoodCredit"]     = has_good_credit
        input_row.loc[:, "IsMarried"]         = is_married
        input_row.loc[:, "CoapplicantIncome"] = svals["CoapplicantIncome"]
        input_row.loc[:, "LoanAmount"]        = svals["LoanAmount"]
        for opt in ["TotalIncome", "EMI", "DTI"]:
            if opt in input_row.columns:
                input_row.loc[:, opt] = svals[opt]

        # One-hot: property area chosen by user -> set the correct dummy to 1
        desired_pa = f"PA_{pa_label}"
        if desired_pa not in input_row.columns and pa_label in AREA_MAP.values():
            pa_code = [k for k, v in AREA_MAP.items() if v == pa_label][0]
            desired_pa = f"PA_{pa_code}"
        if desired_pa in input_row.columns:
            input_row.loc[:, desired_pa] = 1

        # One-hot: dependents. Map display label "3+" back to 3 for the dummy column.
        dep_numeric = 3 if dep_choice == "3+" else int(dep_choice)
        dep_col = f"Dep_{dep_numeric}"
        if dep_col in input_row.columns:
            input_row.loc[:, dep_col] = 1

        # Predict
        if st.button("Predict with Naive Bayes", key="btn_nb_predict_home"):
            prob = nb_model.predict_proba(input_row)[0, 1]
            pred = int(prob >= th_nb)
            label = "Approved" if pred == 1 else "Not Approved"

            st.success(f"**Prediction:** {label}  |  **P(Approved)** = {prob:.2f}  |  **Threshold** = {th_nb:.2f}")

            # Probability gauge
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': '%'},
                title={'text': "Probability of Approval"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'threshold': {'line': {'width': 3}, 'thickness': 0.75, 'value': th_nb * 100}
                }
            ))
            st.plotly_chart(gauge, use_container_width=True)


# =========================
# Streamlit Page: Interpretation & Conclusions
# =========================
def page6():
    """
    Side-by-side comparison of DT and NB using 10-fold OOF predictions.
    Also shows feature importance (DT native; NB via permutation or correlation fallback).
    """
    st.subheader("🧾 Interpretation & Conclusions")

    # Guards for required frames
    if "LoanData" not in globals():
        st.error("`LoanData` not found.")
        st.stop()
    if "LoanData_scaled" not in globals():
        st.error("`LoanData_scaled` not found. Please build it in preprocessing.")
        st.stop()

    # Use the features currently set for DT (default or from DT page)
    dt_features = st.session_state.get("selected_features", dt_features_default)
    X_dt = LoanData[dt_features].copy()
    y_dt = LoanData["Target"].copy()

    # Ensure the DT model matches current features
    dt_model = st.session_state.get("dt_model", model)
    dt_model = ensure_dt_model(dt_model, X_dt, y_dt)
    st.session_state["dt_model"] = dt_model

    dt_threshold = float(st.session_state.get("dt_threshold", 0.50))

    # NB objects from session (if the user has visited the NB page)
    nb_model = st.session_state.get("nb_model", None)
    nb_threshold = float(st.session_state.get("nb_threshold", 0.50))
    nb_features = st.session_state.get("nb_features", None)

    if nb_model is None or nb_features is None:
        st.warning("Naive Bayes model or features not found in session. Open the 'Classification: Naive Bayes' page first.")

    if nb_model is not None and nb_features is not None:
        missing_nb = [c for c in nb_features + ["Target"] if c not in LoanData_scaled.columns]
        if missing_nb:
            st.error(f"Missing columns for Naive Bayes in LoanData_scaled: {missing_nb}")
            st.stop()
        X_nb = LoanData_scaled[nb_features].copy()
        y_nb = LoanData_scaled["Target"].copy()

    # Consistent 10-fold CV for both models
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Helper: robust OOF predict_proba (works for wrapped models too)
    def oof_predict_proba(estimator, X, y, cv):
        if hasattr(estimator, "get_params"):
            try:
                return cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1]
            except Exception:
                pass
        # Fallback manual loop if cross_val_predict can't handle it
        p = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in cv.split(X, y):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr = y.iloc[train_idx]
            est = estimator
            if hasattr(estimator, "get_params"):
                est = clone(estimator)
            est.fit(X_tr, y_tr)
            p[test_idx] = est.predict_proba(X_te)[:, 1]
        return p

    # DT OOF predictions & metrics
    dt_prob_cv = oof_predict_proba(dt_model, X_dt, y_dt, cv)
    dt_pred_cv = (dt_prob_cv >= dt_threshold).astype(int)
    dt_metrics = {
        "Model": "Decision Tree",
        "Accuracy": accuracy_score(y_dt, dt_pred_cv),
        "Precision": precision_score(y_dt, dt_pred_cv, zero_division=0),
        "Recall": recall_score(y_dt, dt_pred_cv, zero_division=0),
        "F1": f1_score(y_dt, dt_pred_cv, zero_division=0),
        "ROC AUC": roc_auc_score(y_dt, dt_prob_cv),
        "Threshold": dt_threshold
    }

    # NB OOF predictions & metrics (if present)
    nb_metrics = None
    if nb_model is not None and nb_features is not None:
        nb_prob_cv = oof_predict_proba(nb_model, X_nb, y_nb, cv)
        nb_pred_cv = (nb_prob_cv >= nb_threshold).astype(int)
        nb_metrics = {
            "Model": "Naive Bayes",
            "Accuracy": accuracy_score(y_nb, nb_pred_cv),
            "Precision": precision_score(y_nb, nb_pred_cv, zero_division=0),
            "Recall": recall_score(y_nb, nb_pred_cv, zero_division=0),
            "F1": f1_score(y_nb, nb_pred_cv, zero_division=0),
            "ROC AUC": roc_auc_score(y_nb, nb_prob_cv),
            "Threshold": nb_threshold
        }

    # Summarize results in a table and a grouped bar chart
    rows = [dt_metrics] + ([nb_metrics] if nb_metrics is not None else [])
    summary_df = pd.DataFrame(rows)
    st.markdown("### 📊 Model Performance Summary (10-Fold CV, Out-of-Fold)")
    st.dataframe(
        summary_df.set_index("Model").style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1": "{:.4f}", "ROC AUC": "{:.4f}", "Threshold": "{:.2f}"
        })
    )
    melted = summary_df.melt(
        id_vars=["Model", "Threshold"],
        value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC AUC"],
        var_name="Metric", value_name="Score"
    )
    fig_bar = px.bar(melted, x="Metric", y="Score", color="Model", barmode="group",
                     title="Model Metrics Comparison (10-Fold CV)")
    fig_bar.update_layout(yaxis=dict(tickformat=".2f"))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # Feature importance
    st.markdown("### 🧠 Which Features Were Most Predictive?")

    # DT native importance (or fallback tiny tree if not available)
    try:
        if hasattr(dt_model, "feature_importances_") and len(dt_model.feature_importances_) == len(dt_features):
            dt_importance = pd.Series(dt_model.feature_importances_, index=dt_features)
        else:
            tmp_tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_dt, y_dt)
            dt_importance = pd.Series(tmp_tree.feature_importances_, index=dt_features)
        dt_importance = dt_importance.sort_values(ascending=False)
        fig_dt_imp = px.bar(dt_importance.iloc[::-1], orientation="h", title="Decision Tree Feature Importance")
        st.plotly_chart(fig_dt_imp, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute Decision Tree importance: {e}")

    # NB permutation importance (or correlation fallback if permutation isn't supported)
    if nb_model is not None and nb_features is not None:
        try:
            nb_fit = clone(nb_model).fit(X_nb, y_nb) if hasattr(nb_model, "get_params") else nb_model
            if hasattr(nb_fit, "fit"):
                nb_fit.fit(X_nb, y_nb)
            perm = permutation_importance(nb_fit, X_nb, y_nb, scoring="f1", n_repeats=10, random_state=42)
            nb_imp = pd.Series(perm.importances_mean, index=nb_features).sort_values(ascending=False)
            top_k = min(12, len(nb_imp))
            fig_nb_imp = px.bar(nb_imp.iloc[:top_k][::-1], orientation="h",
                                title=f"Naive Bayes Permutation Importance (Top {top_k})")
            st.plotly_chart(fig_nb_imp, use_container_width=True)
        except Exception:
            # Fallback: absolute correlation with Target (proxy signal)
            st.info("Permutation importance for NB not available; showing target correlations instead.")
            nb_corr = pd.concat([X_nb.select_dtypes(include=[np.number]), y_nb], axis=1).corr()["Target"].drop("Target")
            nb_imp = nb_corr.abs().sort_values(ascending=False)
            top_k = min(12, len(nb_imp))
            fig_nb_corr = px.bar(nb_imp.iloc[:top_k][::-1], orientation="h",
                                 title=f"Naive Bayes (proxy) | |corr(feature, Target)| (Top {top_k})")
            st.plotly_chart(fig_nb_corr, use_container_width=True)

    st.divider()

    # Brief narrative recap
    st.markdown("### 🧾 Summary & Takeaways")
    dt_top = []
    try:
        dt_top = [f for f, v in dt_importance.items() if v > 0][:3]
    except Exception:
        pass

    nb_top = []
    try:
        nb_top = nb_imp.index[:3].tolist()
    except Exception:
        pass

    def fmt(m, k):
        return f"{m[k]:.3f}" if (m and k in m) else "—"

    st.markdown(
        f"""
- **Most predictive features (Decision Tree):** {", ".join(dt_top) if dt_top else "—"}  
- **Most predictive features (Naive Bayes):** {", ".join(nb_top) if nb_top else "—"}  

**Performance (10-fold CV, OOF):**
- Decision Tree — Accuracy: **{fmt(dt_metrics, 'Accuracy')}**, Precision: **{fmt(dt_metrics, 'Precision')}**, Recall: **{fmt(dt_metrics, 'Recall')}**, F1: **{fmt(dt_metrics, 'F1')}**
- Naive Bayes — Accuracy: **{fmt(nb_metrics, 'Accuracy')}**, Precision: **{fmt(nb_metrics, 'Precision')}**, Recall: **{fmt(nb_metrics, 'Recall')}**, F1: **{fmt(nb_metrics, 'F1')}**

**Trade-offs:**
- The **Decision Tree** is interpretable with clear splits; feature importances highlight the drivers.
- **Naive Bayes** is simple and robust; with one-hot + scaling and engineered features it’s competitive.
- Adjust **thresholds** to trade precision vs recall. Current thresholds — DT: **{dt_threshold:.2f}**, NB: **{nb_threshold:.2f}**.
"""
    )


# =========================
# App Router
# =========================
# Single select control on the sidebar to choose which page to show.
pages = {
    "Dataset": page1,
    "Exploratory Data Analysis": page2,
    "Classification: Decision Tree": page3,
    "Classification: Naive Bayes": page4,
    "Prediction": page5,
    "Interpretation & Conclusions": page6
}

selectpage = st.sidebar.selectbox("Select a Page", list(pages.keys()))
pages[selectpage]()  # Render the chosen page
