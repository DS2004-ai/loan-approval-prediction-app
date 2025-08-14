# Core Libraries
import pandas as pd
import numpy as np

# Visualization
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# Modeling & Evaluation
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, roc_auc_score
)

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import os

# Get current script directory
BASE_DIR = os.path.dirname(__file__)

# Path to CSV inside your project folder
file_path = os.path.join(BASE_DIR, "Loan approval.csv")

# Load CSV
RawData = pd.read_csv(file_path)


# 2. Create a copy for processing
LoanData = RawData.copy()

# 3. Drop Loan_ID (irrelevant for prediction)
if "Loan_ID" in LoanData.columns:
    LoanData.drop("Loan_ID", axis=1, inplace=True)

# 4. Handle missing values
# Categorical → Most frequent
cat_cols = LoanData.select_dtypes(include="object").columns
cat_imputer = SimpleImputer(strategy="most_frequent")
LoanData[cat_cols] = cat_imputer.fit_transform(LoanData[cat_cols])

# Numerical → Median
num_cols = LoanData.select_dtypes(include=["int64", "float64"]).columns
num_imputer = SimpleImputer(strategy="median")
LoanData[num_cols] = num_imputer.fit_transform(LoanData[num_cols])

# 5. Encode target variable
LoanData["Target"] = LoanData["Loan_Status"].map({"Y": 1, "N": 0})
LoanData.drop("Loan_Status", axis=1, inplace=True)

# 6. Encode binary categorical features
label_encoders = {}
binary_cols = ["Gender", "Married", "Education", "Self_Employed"]
for col in binary_cols:
    if col in LoanData.columns:
        le = LabelEncoder()
        LoanData[col] = le.fit_transform(LoanData[col])
        label_encoders[col] = le

# Clean Dependents -> int
if "Dependents" in LoanData.columns:
    LoanData["Dependents"] = LoanData["Dependents"].astype(str).str.replace("+", "", regex=False)
    LoanData["Dependents"] = pd.to_numeric(LoanData["Dependents"], errors="coerce").fillna(0).astype(int)

# Label encode Property_Area if it's object
if "Property_Area" in LoanData.columns and LoanData["Property_Area"].dtype == "object":
    le = LabelEncoder()
    LoanData["Property_Area"] = le.fit_transform(LoanData["Property_Area"])

# 8. IQR-based Outlier Capping
def iqr_cap_series(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return series.clip(lower, upper)

numeric_to_cap = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
for col in numeric_to_cap:
    if col in LoanData.columns:
        LoanData[col] = iqr_cap_series(LoanData[col], factor=1.5)

# 9. Rename columns for clarity
LoanData.rename(columns={
    "Gender": "IsMale",
    "Married": "IsMarried",
    "Education": "IsGraduate",
    "Self_Employed": "IsSelfEmployed",
    "Credit_History": "HasGoodCredit"
}, inplace=True)

# === Feature engineering (before scaling / OHE) ===
LoanData["TotalIncome"] = LoanData["ApplicantIncome"] + LoanData["CoapplicantIncome"]

term_safe = LoanData["Loan_Amount_Term"].replace(0, np.nan)
term_safe = term_safe.fillna(term_safe.median())
emi = (LoanData["LoanAmount"] / term_safe).replace([np.inf, -np.inf], np.nan).fillna(0)
LoanData["EMI"] = emi

ti_safe = LoanData["TotalIncome"].replace(0, np.nan).fillna(LoanData["TotalIncome"].median())
LoanData["DTI"] = (LoanData["EMI"] / ti_safe).replace([np.inf, -np.inf], np.nan).fillna(0)

# 2) Columns to scale (extended for NB)
cols_to_scale = [
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'TotalIncome', 'EMI', 'DTI'
]

# 3) Scale
scaler = MinMaxScaler(feature_range=(0, 1))
LoanData_scaled = LoanData.copy()
LoanData_scaled.loc[:, cols_to_scale] = scaler.fit_transform(LoanData[cols_to_scale])

# --- One-hot encode directly on LoanData_scaled (keep all levels) ---
OHE_COLS = []
if "Property_Area" in LoanData_scaled.columns:
    OHE_COLS.append("Property_Area")
if "Dependents" in LoanData_scaled.columns:
    OHE_COLS.append("Dependents")

if OHE_COLS:
    already_ohe = any(c.startswith("PA_") or c.startswith("Dep_") for c in LoanData_scaled.columns)
    if not already_ohe:
        LoanData_scaled = pd.get_dummies(
            LoanData_scaled,
            columns=OHE_COLS,
            prefix=["PA" if c == "Property_Area" else "Dep" for c in OHE_COLS],
            drop_first=False  # keep all categories e.g., Dep_0 present
        )

# find dummy columns and cast to 0/1
dummy_cols = [c for c in LoanData_scaled.columns if c.startswith("PA_") or c.startswith("Dep_")]
if dummy_cols:
    LoanData_scaled[dummy_cols] = LoanData_scaled[dummy_cols].astype(int)

# keep Target at the end
if "Target" in LoanData_scaled.columns:
    cols = [c for c in LoanData_scaled.columns if c != "Target"] + ["Target"]
    LoanData_scaled = LoanData_scaled[cols]

def build_nb_features(df):
    base = ["HasGoodCredit", "CoapplicantIncome", "LoanAmount", "IsMarried",
            "TotalIncome", "EMI", "DTI"]
    ohe = [c for c in df.columns if c.startswith("PA_") or c.startswith("Dep_")]
    return base + ohe

NB_FEATURES = build_nb_features(LoanData_scaled)

# Human map for Property Area (used for UI fallbacks)
area_map = {0: "Rural", 1: "Semiurban", 2: "Urban"}

# === Global Model Training (Decision Tree default) ===
features = ["HasGoodCredit", "CoapplicantIncome", "LoanAmount", "Property_Area", "IsMarried"]
target = "Target"

X = LoanData[features]
y = LoanData[target]

# Default thresholds (will be overwritten by pages)
dt_default_threshold = 0.50
nb_default_threshold = 0.50

# Base model for page1
model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=1,
    min_samples_split=2,
    criterion="gini",
    random_state=42
)
model.fit(X, y)

# =========================
# Streamlit Pages
# =========================
def page1():
    st.subheader("🔮 Quick Prediction")

    # --- Pull thresholds saved from other pages (with safe fallbacks) ---
    th_dt = float(st.session_state.get("dt_threshold", dt_default_threshold))
    th_nb = float(st.session_state.get("nb_threshold", nb_default_threshold))

    st.info(
        f"**Current thresholds** — "
        f"Decision Tree: `{th_dt:.2f}`  |  Naive Bayes: `{th_nb:.2f}`\n\n"
        f"_Tip: tune DT on the Decision Tree page and NB on the Naive Bayes page._"
    )

    model_choice = st.selectbox(
        "Choose model",
        ["Decision Tree", "Naive Bayes"],
        index=0,
        key="home_model_choice"
    )

    # =========================
    # Decision Tree Prediction
    # =========================
    if model_choice == "Decision Tree":
        st.markdown("#### 🌳 Decision Tree (raw inputs)")

        dt_features = ["HasGoodCredit", "CoapplicantIncome", "LoanAmount", "Property_Area", "IsMarried"]
        missing_dt = [c for c in dt_features + ["Target"] if c not in LoanData.columns]
        if missing_dt:
            st.error(f"Missing columns in LoanData for Decision Tree: {missing_dt}")
            st.stop()

        dt_model = model if isinstance(model, DecisionTreeClassifier) else DecisionTreeClassifier(
            max_depth=3, random_state=42).fit(LoanData[dt_features], LoanData["Target"])

        cols = st.columns(2)
        med = LoanData[dt_features].median(numeric_only=True)

        area_map = {0: "Rural", 1: "Semiurban", 2: "Urban"}
        with cols[0]:
            has_good_credit = st.selectbox("Has Good Credit? (1=Yes, 0=No)", [1, 0], index=0, key="dt_good_credit")
            is_married      = st.selectbox("Is Married? (1=Yes, 0=No)", [1, 0], index=0, key="dt_is_married")
            pa_label_ui     = st.selectbox("Property Area", list(area_map.values()), index=1, key="dt_property_area")
            property_area   = [k for k, v in area_map.items() if v == pa_label_ui][0]

        with cols[1]:
            coapp_income = st.number_input("Coapplicant Income (raw)", value=float(med.get("CoapplicantIncome", 0.0)),
                                           step=100.0, key="dt_coapp_income")
            loan_amount  = st.number_input("Loan Amount (raw)", value=float(med.get("LoanAmount", 100.0)),
                                           step=1.0, key="dt_loan_amount")

        X_pred = pd.DataFrame([[has_good_credit, coapp_income, loan_amount, property_area, is_married]],
                              columns=dt_features)

        if st.button("Predict with Decision Tree", key="btn_dt_predict"):
            prob = dt_model.predict_proba(X_pred)[0, 1]
            pred = int(prob >= th_dt)
            label = "Approved" if pred == 1 else "Not Approved"

            st.success(f"**Prediction:** {label}  |  **P(Approved)** = {prob:.2f}  |  **Threshold** = {th_dt:.2f}")

            # Gauge
            import plotly.graph_objects as go
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

    # ======================
    # Naive Bayes Prediction
    # ======================
    else:
        st.markdown("#### 🧠 Naive Bayes (raw inputs → auto‑scaled)")

        if "nb_model" not in st.session_state:
            st.warning("Naive Bayes model not trained yet — open 'Classification: Naive Bayes' page first.")
            st.stop()

        nb_model    = st.session_state["nb_model"]
        nb_features = st.session_state["nb_features"]
        pa_cols     = st.session_state["nb_pa_cols"]
        dep_cols    = st.session_state["nb_dep_cols"]

        # derive labels from PA_* columns (numeric or labeled suffixes)
        area_map = {0: "Rural", 1: "Semiurban", 2: "Urban"}
        def _pa_human_labels(pa_cols):
            if not pa_cols:
                return list(area_map.values())
            suffixes = [c.split("PA_", 1)[1] for c in pa_cols]
            labels = []
            for s in sorted(suffixes, key=lambda x: str(x)):
                try: labels.append(area_map[int(s)])
                except: labels.append(s)
            return labels

        def _dep_values(dep_cols):
            if not dep_cols: return [0, 1, 2, 3]
            return sorted(int(c.split("Dep_", 1)[1]) for c in dep_cols)

        pa_labels_from_cols = _pa_human_labels(pa_cols)
        dep_values = _dep_values(dep_cols)

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
            dependents = st.selectbox("Number of Dependents", dep_values, index=0, key="nb_dependents_home")

        # Scale & engineer to match training
        def scale_and_engineer(coapp_raw, loan_raw):
            appl_med = float(LoanData["ApplicantIncome"].median())
            term_med = float(LoanData["Loan_Amount_Term"].replace(0, np.nan).dropna().median())
            total_income = appl_med + coapp_raw
            emi = 0.0 if term_med == 0 else loan_raw / term_med
            dti = 0.0 if total_income == 0 else emi / total_income
            row = pd.DataFrame([[appl_med, coapp_raw, loan_raw, total_income, emi, dti]],
                               columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                                        'TotalIncome', 'EMI', 'DTI'])
            scaled = scaler.transform(row)
            return dict(zip(row.columns, scaled.ravel()))

        svals = scale_and_engineer(coapp_raw, loan_raw)

        # Build aligned input row
        input_row = pd.DataFrame(np.zeros((1, len(nb_features))), columns=nb_features)
        input_row.loc[:, "HasGoodCredit"]     = has_good_credit
        input_row.loc[:, "IsMarried"]         = is_married
        input_row.loc[:, "CoapplicantIncome"] = svals["CoapplicantIncome"]
        input_row.loc[:, "LoanAmount"]        = svals["LoanAmount"]
        if "TotalIncome" in input_row.columns: input_row.loc[:, "TotalIncome"] = svals["TotalIncome"]
        if "EMI" in input_row.columns:         input_row.loc[:, "EMI"] = svals["EMI"]
        if "DTI" in input_row.columns:         input_row.loc[:, "DTI"] = svals["DTI"]

        # One-hot flags
        desired_pa = f"PA_{pa_label}"
        if desired_pa not in input_row.columns and pa_label in area_map.values():
            pa_code = [k for k, v in area_map.items() if v == pa_label][0]
            desired_pa = f"PA_{pa_code}"
        if desired_pa in input_row.columns:
            input_row.loc[:, desired_pa] = 1

        dep_col = f"Dep_{dependents}"
        if dep_col in input_row.columns:
            input_row.loc[:, dep_col] = 1

        if st.button("Predict with Naive Bayes", key="btn_nb_predict_home"):
            prob = nb_model.predict_proba(input_row)[0, 1]
            pred = int(prob >= th_nb)
            label = "Approved" if pred == 1 else "Not Approved"

            st.success(f"**Prediction:** {label}  |  **P(Approved)** = {prob:.2f}  |  **Threshold** = {th_nb:.2f}")

            # Gauge
            import plotly.graph_objects as go
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



def page2():
    st.subheader("📄 Loan Approval Datasets")

    if st.checkbox("Raw Loan Approval Data", key="chk_raw"):
        st.write(RawData)

    if st.checkbox("IQR + LE + FE Loan Approval Data", key="chk_proc"):
        st.write(LoanData)

    if st.checkbox("Scaled + One-Hot Loan Approval Data", key="chk_scaled"):
        st.write(LoanData_scaled)


def page3():
    st.subheader("📊 Exploratory Data Analysis (Plotly, RawData)")

    # --- Work on a safe copy ---
    raw_df = RawData.copy()

    # --- Column typing ---
    num_cols = raw_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = raw_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # --- Quick KPIs ---
    c1, c2, c3, c4 = st.columns(4)
    total_rows = len(raw_df)
    approved = raw_df["Loan_Status"].eq("Y").sum() if "Loan_Status" in raw_df.columns else None
    not_approved = raw_df["Loan_Status"].eq("N").sum() if "Loan_Status" in raw_df.columns else None
    med_income = raw_df["ApplicantIncome"].median() if "ApplicantIncome" in raw_df.columns else None

    with c1:
        st.metric("Total Records", f"{total_rows:,}")
    with c2:
        st.metric("Approved (Y)", f"{approved:,}" if approved is not None else "—")
    with c3:
        st.metric("Not Approved (N)", f"{not_approved:,}" if not_approved is not None else "—")
    with c4:
        st.metric("Median Applicant Income", f"{med_income:,.0f}" if med_income is not None else "—")

    st.divider()

    # --- Histograms for numeric features ---
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

    # --- Correlation heatmap (numeric) ---
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

    # --- High-value visuals for this dataset ---
    st.markdown("### 🧭 Additional Visuals")

    # Loan Status counts (version‑proof)
    if "Loan_Status" in raw_df.columns:
        loan_status_counts = (
            raw_df["Loan_Status"]
            .value_counts(dropna=False)
            .rename_axis("Loan_Status")
            .reset_index(name="Count")
        )

        fig = px.bar(
            loan_status_counts,
            x="Loan_Status",
            y="Count",
            text_auto=True,
            title="Loan Status Counts"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Property Area by Loan Status
    if {"Property_Area", "Loan_Status"}.issubset(raw_df.columns):
        fig = px.histogram(raw_df, x="Property_Area", color="Loan_Status",
                           barmode="group", text_auto=True,
                           title="Loan Status by Property Area")
        st.plotly_chart(fig, use_container_width=True)

    # Income vs Loan Amount
    if {"ApplicantIncome", "LoanAmount"}.issubset(raw_df.columns):
        color_col = "Loan_Status" if "Loan_Status" in raw_df.columns else None
        fig = px.scatter(raw_df, x="ApplicantIncome", y="LoanAmount",
                         color=color_col, hover_data=raw_df.columns,
                         title="Applicant Income vs Loan Amount")
        st.plotly_chart(fig, use_container_width=True)

    # Box plots: LoanAmount by categorical drivers (if present)
    if {"LoanAmount", "Education"}.issubset(raw_df.columns):
        fig = px.box(raw_df, x="Education", y="LoanAmount", color="Education",
                     title="Loan Amount by Education")
        st.plotly_chart(fig, use_container_width=True)

    if {"LoanAmount", "Property_Area"}.issubset(raw_df.columns):
        fig = px.box(raw_df, x="Property_Area", y="LoanAmount", color="Property_Area",
                     title="Loan Amount by Property Area")
        st.plotly_chart(fig, use_container_width=True)

    # Credit History impact
    if {"Credit_History", "Loan_Status"}.issubset(raw_df.columns):
        fig = px.histogram(raw_df, x="Credit_History", color="Loan_Status",
                           barmode="group", text_auto=True,
                           title="Loan Status by Credit History")
        st.plotly_chart(fig, use_container_width=True)

def page4():
    global model  # so updates carry over to page1

    st.subheader("🌳 Decision Tree — 10-Fold Cross-Validation (With GridSearch)")

    features = ["HasGoodCredit", "CoapplicantIncome", "LoanAmount", "Property_Area", "IsMarried"]
    target = "Target"
    X = LoanData[features]
    y = LoanData[target]

    st.sidebar.header("🛠️ Model Options")
    use_grid = st.sidebar.checkbox("Use GridSearchCV", value=True, key="dt_use_grid")
    refit_metric = st.sidebar.selectbox("Optimize (refit) for",
                                        options=["f1", "precision", "recall", "accuracy"], index=0, key="dt_refit")
    st.sidebar.markdown("### 🎚️ Decision Threshold (Tree)")
    dt_threshold = st.sidebar.slider("Approve if P(Approved) ≥", 0.10, 0.95, 0.50, 0.01, key="dt_threshold_slider")
    st.session_state["dt_threshold"] = float(dt_threshold)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

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
        model = grid.best_estimator_

        st.write("#### ✅ Best hyperparameters:")
        st.write(grid.best_params_)
        st.write(f"Best mean CV {refit_metric}: {grid.best_score_:.4f}")
    else:
        max_depth = st.sidebar.slider("Max Depth", 1, 10, 3, key="dt_max_depth")
        min_samples_leaf = st.sidebar.slider("Min Samples per Leaf", 1, 50, 10, key="dt_min_leaf")
        min_samples_split = st.sidebar.slider("Min Samples to Split", 2, 100, 10, key="dt_min_split")
        criterion = st.sidebar.selectbox("Split Criterion", ["gini", "entropy"], key="dt_criterion")

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=42
        )

    # CV predictions for metrics
    prob_cv = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred_cv = (prob_cv >= dt_threshold).astype(int)

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

    cm = confusion_matrix(y, y_pred_cv)
    st.markdown("### 🧮 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=["Not Approved", "Approved"],
                yticklabels=["Not Approved", "Approved"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    fpr, tpr, _ = roc_curve(y, prob_cv)
    st.markdown("### 📈 ROC Curve (10‑Fold CV Probabilities)")
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # Train final model on ALL data (for page1)
    model.fit(X, y)
    st.success("✅ Tree updated — available on the Dataset/Prediction page.")

    # Keep the selected features & *DT* threshold for other pages
    st.session_state["selected_features"] = features
    # keep a generic 'threshold' only if some legacy code still reads it
    st.session_state["threshold"] = float(dt_threshold)
    st.session_state["dt_threshold"] = float(dt_threshold)  # <-- the one page1 reads for DT

    st.markdown("### 🌳 Tree Structure (trained on all data)")
    fig_tree, ax_tree = plt.subplots(figsize=(12, 6))
    plot_tree(model, feature_names=features, class_names=["Not Approved", "Approved"],
              filled=True, rounded=True, fontsize=10, ax=ax_tree)
    st.pyplot(fig_tree)


def page5():
    st.subheader("🧠 Naive Bayes — 10‑Fold CV (Tuned, Multiple Variants)")

    # -- Use globally prepared, scaled & one-hot encoded dataframe --
    if "LoanData_scaled" not in globals():
        st.error("`LoanData_scaled` not found. Please create it before using this page.")
        st.stop()
    df = LoanData_scaled

    # ==== Sidebar controls ====
    st.sidebar.header("🛠️ Naive Bayes Options")
    optimize_for = st.sidebar.selectbox(
        "Optimize threshold for",
        ["accuracy", "f1", "precision", "recall"],
        index=0,
        key="nb_opt_metric"
    )
    use_categorical_nb = st.sidebar.checkbox("Try CategoricalNB (discretize features)", value=False, key="nb_use_cat")

    # ---- Build feature list dynamically ----
    base_feats = ["HasGoodCredit", "CoapplicantIncome", "LoanAmount", "IsMarried",
                  "TotalIncome", "EMI", "DTI"]


    pa_cols  = [c for c in df.columns if c.startswith("PA_")]
    dep_cols = [c for c in df.columns if c.startswith("Dep_")]
    features = base_feats + pa_cols + dep_cols
    target = "Target"

    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        st.error(f"Missing columns in LoanData_scaled: {missing}")
        st.stop()

    X = df[features].copy()
    y = df[target]
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # ==== Helper: compute metric at threshold ====
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
        else:
            return accuracy_score(y_true, yp)

    # ==== Variant A: GaussianNB (tune var_smoothing) ====
    if not use_categorical_nb:
        param_grid = {"var_smoothing": np.logspace(-13, -5, 17)}
        nb_grid = GridSearchCV(
            GaussianNB(),
            param_grid=param_grid,
            scoring="accuracy",      # model selection metric (kept as accuracy)
            cv=cv,
            n_jobs=-1
        )
        nb_grid.fit(X, y)
        best_nb = nb_grid.best_estimator_
        st.write(f"🔧 Best var_smoothing: **{nb_grid.best_params_['var_smoothing']}**")

        # Out-of-fold probabilities for threshold search
        prob_cv = cross_val_predict(best_nb, X, y, cv=cv, method="predict_proba")[:, 1]

    # ==== Variant B: CategoricalNB (discretize continuous features) ====
    else:
        # Discretize continuous columns into bins then use CategoricalNB
        from sklearn.naive_bayes import CategoricalNB
        from sklearn.preprocessing import KBinsDiscretizer
        # continuous-like columns among our features:
        cont = [c for c in ["CoapplicantIncome","LoanAmount","TotalIncome","EMI","DTI"] if c in X.columns]
        cat  = [c for c in X.columns if c not in cont]

        # Build discretizer (same across folds to keep deterministic bins)
        kb = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
        X_cont = kb.fit_transform(X[cont]) if cont else np.empty((len(X),0))
        X_cat  = X[cat].to_numpy() if cat else np.empty((len(X),0))
        X_disc = np.hstack([X_cont, X_cat])

        # CategoricalNB expects non-negative integers (we have ordinal bins + 0/1 dummies)
        cat_nb = CategoricalNB(alpha=1.0)  # Laplace smoothing
        # OOF predicted probabilities with a fixed transform inside CV (ok for what-if)
        prob_cv = cross_val_predict(cat_nb, X_disc, y, cv=cv, method="predict_proba")[:, 1]
        best_nb = cat_nb  # for final fit below
        st.info("Using CategoricalNB with 5-bin quantile discretization for continuous features.")

    # ==== Choose threshold that maximizes the chosen metric ====
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = [metric_at_threshold(y, prob_cv, t, optimize_for) for t in thresholds]
    best_idx = int(np.argmax(scores))
    best_t = float(thresholds[best_idx])
    st.session_state["nb_threshold"] = float(best_t)

    # Final CV metrics at that threshold
    y_pred_cv = (prob_cv >= best_t).astype(int)
    acc = accuracy_score(y, y_pred_cv)
    prec = precision_score(y, y_pred_cv, zero_division=0)
    rec = recall_score(y, y_pred_cv, zero_division=0)
    f1  = f1_score(y, y_pred_cv, zero_division=0)
    auc = roc_auc_score(y, prob_cv)

    st.markdown("### 📏 Cross‑Validated Metrics (10‑Fold, Out‑of‑Fold)")
    st.write(f"**Optimized for:** `{optimize_for}`  |  **Best threshold:** `{best_t:.2f}`")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.write(f"**Precision:** {prec:.4f}")
    st.write(f"**Recall:** {rec:.4f}")
    st.write(f"**F1‑score:** {f1:.4f}")
    st.write(f"**ROC AUC:** {auc:.4f}")
    st.caption(f"Features ({len(features)}): {features[:8]}{' ...' if len(features)>8 else ''}")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred_cv)
    st.markdown("### 🧮 Confusion Matrix (10‑Fold CV Predictions)")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=["Not Approved", "Approved"],
                yticklabels=["Not Approved", "Approved"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # ROC
    fpr, tpr, _ = roc_curve(y, prob_cv)
    st.markdown("### 📈 ROC Curve (10‑Fold CV Probabilities)")
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # ==== Fit final NB on ALL data for Page 1 ====
    if not use_categorical_nb:
        best_nb.fit(X, y)
        st.session_state["nb_model"] = best_nb
        st.session_state["nb_features"] = features
        st.session_state["nb_pa_cols"] = pa_cols
        st.session_state["nb_dep_cols"] = dep_cols
    else:
        # Save a small wrapper so Page1 can use it
        class CatNBWrapper:
            def __init__(self, kb, clf, cont_cols, all_features):
                self.kb = kb
                self.clf = clf
                self.cont_cols = cont_cols
                self.all_features = all_features
            def fit(self, X, y):
                Xc = self.kb.fit_transform(X[self.cont_cols]) if self.cont_cols else np.empty((len(X),0))
                Xd = X[[c for c in self.all_features if c not in self.cont_cols]].to_numpy()
                self.clf.fit(np.hstack([Xc, Xd]), y)
                return self
            def predict_proba(self, X):
                Xc = self.kb.transform(X[self.cont_cols]) if self.cont_cols else np.empty((len(X),0))
                Xd = X[[c for c in self.all_features if c not in self.cont_cols]].to_numpy()
                return self.clf.predict_proba(np.hstack([Xc, Xd]))

        cont_cols = [c for c in ["CoapplicantIncome","LoanAmount","TotalIncome","EMI","DTI"] if c in X.columns]
        nb_wrap = CatNBWrapper(
            kb=KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile"),
            clf=CategoricalNB(alpha=1.0),
            cont_cols=cont_cols,
            all_features=features
        ).fit(X, y)

        st.session_state["nb_model"] = nb_wrap
        st.session_state["nb_features"] = features
        st.session_state["nb_pa_cols"] = pa_cols
        st.session_state["nb_dep_cols"] = dep_cols

    st.success("✅ Naive Bayes updated — available on the Dataset/Prediction page.")


def page6():
    st.subheader("🧾 Interpretation & Conclusions")

    # --- Guards & data access ---
    if "LoanData" not in globals():
        st.error("`LoanData` not found.")
        st.stop()
    if "LoanData_scaled" not in globals():
        st.error("`LoanData_scaled` not found. Please build it in preprocessing.")
        st.stop()

    # === Retrieve models, thresholds, and features from session ===
    # Decision Tree
    dt_model = st.session_state.get("dt_model", None)
    dt_threshold = float(st.session_state.get("dt_threshold", 0.50))
    dt_features = st.session_state.get("selected_features",
                                       ["HasGoodCredit", "CoapplicantIncome", "LoanAmount", "Property_Area", "IsMarried"])

    # Fallback to global 'model' if dt_model wasn't saved
    if dt_model is None:
        if 'model' in globals() and isinstance(model, DecisionTreeClassifier):
            dt_model = model
        else:
            dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)

    # Naive Bayes
    nb_model = st.session_state.get("nb_model", None)
    nb_threshold = float(st.session_state.get("nb_threshold", 0.50))
    nb_features = st.session_state.get("nb_features", None)

    if nb_model is None or nb_features is None:
        st.warning("Naive Bayes model or features not found in session. "
                   "Open the 'Classification: Naive Bayes' page first.")
        # We'll still compute DT results; NB will be skipped.

    # === Build matrices ===
    # Decision Tree uses the label-encoded (unscaled) LoanData
    missing_dt = [c for c in dt_features + ["Target"] if c not in LoanData.columns]
    if missing_dt:
        st.error(f"Missing columns for Decision Tree: {missing_dt}")
        st.stop()
    X_dt = LoanData[dt_features].copy()
    y_dt = LoanData["Target"].copy()

    # Naive Bayes uses scaled + one-hot encoded data
    if nb_model is not None and nb_features is not None:
        missing_nb = [c for c in nb_features + ["Target"] if c not in LoanData_scaled.columns]
        if missing_nb:
            st.error(f"Missing columns for Naive Bayes in LoanData_scaled: {missing_nb}")
            st.stop()
        X_nb = LoanData_scaled[nb_features].copy()
        y_nb = LoanData_scaled["Target"].copy()

    # === Helper: OOF probs that works with sklearn models and wrappers ===
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    def oof_predict_proba(estimator, X, y, cv):
        """
        Return out-of-fold class-1 probabilities. Uses cross_val_predict when possible;
        otherwise falls back to a manual CV loop (for wrappers without get_params).
        """
        # Try cross_val_predict (requires sklearn-style estimator with get_params)
        if hasattr(estimator, "get_params"):
            try:
                return cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")[:, 1]
            except Exception:
                pass

        # Manual loop fallback
        import numpy as np
        p = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in cv.split(X, y):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr = y.iloc[train_idx]
            # Clone if possible; else fit the same object (OK for evaluation)
            est = estimator
            if hasattr(estimator, "get_params"):
                from sklearn.base import clone
                est = clone(estimator)
            est.fit(X_tr, y_tr)
            p[test_idx] = est.predict_proba(X_te)[:, 1]
        return p

    # === Compute metrics (10-fold CV, out-of-fold) ===
    # Decision Tree
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

    # Naive Bayes (only if available)
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

    # === Summary table ===
    rows = [dt_metrics] + ([nb_metrics] if nb_metrics is not None else [])
    summary_df = pd.DataFrame(rows)
    st.markdown("### 📊 Model Performance Summary (10‑Fold CV, Out‑of‑Fold)")
    st.dataframe(
        summary_df.set_index("Model").style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1": "{:.4f}", "ROC AUC": "{:.4f}", "Threshold": "{:.2f}"
        })
    )

    # Side-by-side metrics bar chart
    melted = summary_df.melt(
        id_vars=["Model", "Threshold"],
        value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC AUC"],
        var_name="Metric", value_name="Score"
    )
    fig_bar = px.bar(melted, x="Metric", y="Score", color="Model", barmode="group",
                     title="Model Metrics Comparison (10‑Fold CV)")
    fig_bar.update_layout(yaxis=dict(tickformat=".2f"))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # === Feature importance ===
    st.markdown("### 🧠 Which Features Were Most Predictive?")

    # Decision Tree: native importances (use the model from session if it has them)
    try:
        if hasattr(dt_model, "feature_importances_") and len(getattr(dt_model, "feature_importances_", [])) == len(dt_features):
            dt_importance = pd.Series(dt_model.feature_importances_, index=dt_features)
        else:
            # Fit a shallow tree for importance if needed
            tmp_tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_dt, y_dt)
            dt_importance = pd.Series(tmp_tree.feature_importances_, index=dt_features)
        dt_importance = dt_importance.sort_values(ascending=False)
        fig_dt_imp = px.bar(dt_importance.iloc[::-1], orientation="h", title="Decision Tree Feature Importance")
        st.plotly_chart(fig_dt_imp, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute Decision Tree importance: {e}")

    # Naive Bayes: permutation importance if possible
    if nb_model is not None and nb_features is not None:
        try:
            # Make sure model is fitted for permutation importance
            # (Use a clone-like refit on all data if supported, else fit in place)
            if hasattr(nb_model, "get_params"):
                from sklearn.base import clone
                nb_fit = clone(nb_model).fit(X_nb, y_nb)
            else:
                nb_fit = nb_model
                if hasattr(nb_fit, "fit"):
                    nb_fit.fit(X_nb, y_nb)
            perm = permutation_importance(nb_fit, X_nb, y_nb, scoring="f1", n_repeats=10, random_state=42)
            nb_imp = pd.Series(perm.importances_mean, index=nb_features).sort_values(ascending=False)
            top_k = min(12, len(nb_imp))
            fig_nb_imp = px.bar(nb_imp.iloc[:top_k][::-1], orientation="h",
                                title=f"Naive Bayes Permutation Importance (Top {top_k})")
            st.plotly_chart(fig_nb_imp, use_container_width=True)
        except Exception as e:
            st.info("Permutation importance for NB not available; showing target correlations instead.")
            # Fallback: absolute correlation with target (numerical cols only)
            nb_corr = pd.concat([X_nb.select_dtypes(include=[np.number]), y_nb], axis=1).corr()["Target"].drop("Target")
            nb_imp = nb_corr.abs().sort_values(ascending=False)
            top_k = min(12, len(nb_imp))
            fig_nb_corr = px.bar(nb_imp.iloc[:top_k][::-1], orientation="h",
                                 title=f"Naive Bayes (proxy) | |corr(feature, Target)| (Top {top_k})")
            st.plotly_chart(fig_nb_corr, use_container_width=True)

    st.divider()

    # === Narrative conclusions ===
    st.markdown("### 🧾 Summary & Takeaways")
    dt_top = []
    try:
        dt_top = [f for f, v in dt_importance.items() if v > 0][:3]
    except:
        pass

    nb_top = []
    try:
        nb_top = nb_imp.index[:3].tolist()
    except:
        pass

    # Compose performance text safely
    def fmt(m, k):
        return f"{m[k]:.3f}" if (m and k in m) else "—"

    st.markdown(
        f"""
- **Most predictive features (Decision Tree):** {", ".join(dt_top) if dt_top else "—"}  
- **Most predictive features (Naive Bayes):** {", ".join(nb_top) if nb_top else "—"}  

**Performance (10‑fold CV, OOF):**
- Decision Tree — Accuracy: **{fmt(dt_metrics, 'Accuracy')}**, Precision: **{fmt(dt_metrics, 'Precision')}**, Recall: **{fmt(dt_metrics, 'Recall')}**, F1: **{fmt(dt_metrics, 'F1')}**
- Naive Bayes — Accuracy: **{fmt(nb_metrics, 'Accuracy')}**, Precision: **{fmt(nb_metrics, 'Precision')}**, Recall: **{fmt(nb_metrics, 'Recall')}**, F1: **{fmt(nb_metrics, 'F1')}**

**Trade‑offs:**
- The **Decision Tree** is interpretable with clear splits; feature importances highlight the drivers.
- **Naive Bayes** is simple and robust; with one‑hot + scaling and engineered features it’s competitive.
- Adjust **thresholds** to trade precision vs recall. Current thresholds — DT: **{dt_threshold:.2f}**, NB: **{nb_threshold:.2f}**.
"""
    )

    st.info("Tip: Tune thresholds and (for NB) try the categorical variant on their pages to change the precision‑recall balance.")



pages = {
    "Prediction": page1,
    "Dataset": page2,
    "Exploratory Data Analysis": page3,
    "Classification: Decision Tree": page4,
    "Classification: Naive Bayes": page5,
    "Interpretation & Conclusions": page6
}

selectpage = st.sidebar.selectbox("Select a Page", list(pages.keys()))
pages[selectpage]()
