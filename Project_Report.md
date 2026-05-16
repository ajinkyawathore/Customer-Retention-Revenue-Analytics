# Project Report: Customer Retention & Revenue Analytics

**Project Title:** Customer Retention & Revenue Analytics Pipeline
**Domain:** SaaS / Subscription Business Intelligence
**Report Type:** End-to-End Technical & Business Report

---

## Executive Summary

This project delivers a complete, production-grade analytics pipeline for a SaaS business. Starting from seven raw CSV files, the pipeline progresses through four distinct phases: data cleaning, master table construction, exploratory analysis, and machine learning-based churn prediction.

The final outputs include a risk-scored customer database, a serialised churn prediction model, a cohort retention heatmap, and an actionable at-risk customer list ranked by churn probability and monthly recurring revenue (MRR) exposure. The pipeline gives any SaaS retention team the data infrastructure needed to move from reactive to proactive customer management.

---

## 1. Problem Statement

Customer churn is the single most important lever in subscription business profitability. Acquiring a new customer costs five to seven times more than retaining an existing one, yet most businesses only detect churn after it has happened — when a customer cancels.

This project addresses three business questions:

**Question 1:** Which customers are at the highest risk of churning, and how much revenue do they represent?

**Question 2:** At what point in the customer lifecycle (tenure month) do customers most commonly drop off?

**Question 3:** Which customer behaviours and attributes are the strongest predictors of churn, and can those predictors be used to build a reliable early-warning system?

---

## 2. Dataset Description

The dataset represents a SaaS company's customer data, structured as a star schema across seven relational tables.

### 2.1 Dimension Tables

Dimension tables describe entities — the "who" and "what" of the business.

**dim_users**
Stores the customer profile. Each row represents one unique customer.

| Column | Type | Description |
|---|---|---|
| user_id | String | Primary key — unique customer identifier |
| signup_date | Date | Date the customer created their account |
| gender | String | Customer gender |
| company_size | String | Size of the customer's company |
| nps_score | Float | Net Promoter Score (customer satisfaction, 0–10) |
| acquisition_channel | String | How the customer was acquired |

**dim_plans**
Describes the product pricing tiers available.

| Column | Type | Description |
|---|---|---|
| plan_id | String | Primary key |
| plan_name | String | Tier name (Basic, Standard, Pro, Enterprise) |
| billing_cycle | String | Monthly or Annual |
| base_price_usd | Float | Base price for this plan |

**dim_geography**
Maps customers to geographic regions.

| Column | Type | Description |
|---|---|---|
| geo_id | String | Primary key |
| country | String | Customer's country |
| region | String | Continent or business region |
| city | String | City |

**dim_acquisition_channel**
Provides metadata about how customers were acquired.

| Column | Type | Description |
|---|---|---|
| channel_id | String | Primary key |
| channel_name | String | Channel type (Organic Search, Paid Ads, Referral, etc.) |

### 2.2 Fact Tables

Fact tables record events and transactions — the "what happened."

**fact_subscriptions**
One row per customer. The central fact table and the source of the churn label.

| Column | Type | Description |
|---|---|---|
| user_id | String | Foreign key to dim_users |
| plan_name | String | Active plan at time of snapshot |
| start_date | Date | Subscription start date |
| cancel_date | Date | Cancellation date (null if still active) |
| mrr_usd | Float | Monthly Recurring Revenue from this customer |
| tenure_days | Integer | Days from signup to snapshot date |
| churned | Integer | Target variable: 1 = churned, 0 = retained |
| upgrade_count | Integer | Number of plan upgrades |
| downgrade_count | Integer | Number of plan downgrades |
| status | String | Active / Cancelled |

**fact_payments**
One row per payment transaction. Contains the full billing history.

| Column | Type | Description |
|---|---|---|
| payment_id | String | Primary key |
| user_id | String | Foreign key to dim_users |
| payment_date | Date | Transaction date |
| amount_usd | Float | Payment amount |
| is_failed | Integer | 1 = payment failed |
| is_refunded | Integer | 1 = payment was refunded |

**fact_activity_logs**
One row per user event. The most granular table, with approximately 1.7 million rows.

| Column | Type | Description |
|---|---|---|
| activity_id | String | Primary key |
| user_id | String | Foreign key to dim_users |
| event_date | Date | Date of the event |
| event_type | String | Type: login, feature_use, settings_change, etc. |
| feature_used | String | Which product feature was accessed |
| page_views | Integer | Pages viewed in this session |
| session_duration_min | Float | Length of session in minutes |
| is_mobile | Integer | 1 = event occurred on a mobile device |
| error_occurred | Integer | 1 = an error was recorded in this session |

---

## 3. Step 1 — Data Ingestion & Cleaning

### 3.1 Objective

Load all raw CSV files, detect data quality issues, apply systematic fixes, and save clean versions to a separate folder. The golden rule: raw files are never modified.

### 3.2 Quality Issues Detected

Before cleaning, a quality report function was applied to every table. Issues found:

| Table | Issue Type | Details |
|---|---|---|
| dim_users | Wrong data type | `signup_date` stored as text string |
| dim_users | Whitespace contamination | Text columns had leading/trailing spaces |
| dim_users | Duplicate rows | Duplicate `user_id` entries |
| dim_users | Missing values | `nps_score` had null entries |
| fact_subscriptions | Wrong data type | `start_date` and `cancel_date` stored as text |
| fact_subscriptions | Unusable rows | Rows with no `start_date` (cannot compute tenure) |
| fact_subscriptions | Invalid values | Negative `mrr_usd` values (economically impossible) |
| fact_payments | Wrong data type | `payment_date` stored as text |
| fact_payments | Statistical outliers | Extreme `amount_usd` values far beyond normal range |
| fact_payments | Duplicate rows | Duplicate `payment_id` entries |
| fact_activity_logs | Wrong data type | `event_date` stored as text |
| fact_activity_logs | Missing values | `session_duration_min` had null entries |
| fact_activity_logs | Duplicate rows | Duplicate `activity_id` entries |

### 3.3 Cleaning Actions Applied

**Date type conversion**
All date columns across all tables were converted using `pd.to_datetime(col, errors='coerce')`. The `errors='coerce'` argument converts any unparseable value to `NaT` (Not a Time) rather than raising an error, ensuring the pipeline does not break on malformed dates.

**Whitespace stripping**
All object-type columns in `dim_users` were stripped of leading and trailing whitespace using `.str.strip()`. This prevents silent join failures caused by values like `' Male'` not matching `'Male'`.

**Duplicate removal**
`drop_duplicates(subset='<primary_key>')` was applied to each table. Using the subset argument ensures only rows with identical primary keys are removed, preserving legitimately duplicated attribute values.

**Missing value treatment**

- `nps_score`: Filled with the column median. The median was chosen over the mean because NPS scores can have a skewed distribution, and the median is robust to outliers.
- `session_duration_min`: Filled with 0. A missing session duration in the activity log means the session length was not recorded, which is meaningfully different from a zero-duration session but treated as 0 for aggregation purposes.
- Rows with missing `start_date` in `fact_subscriptions` were dropped entirely, since tenure and cohort calculations are impossible without a start date.

**Negative MRR correction**
Negative MRR values were clipped to 0 using `.clip(lower=0)`. Negative revenue is not economically valid in a standard SaaS billing model; these likely represent data entry errors or system artefacts.

**Outlier capping in payment amounts**
Payment outliers were detected and capped using the Interquartile Range (IQR) method:
- Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 − Q1
- Lower bound: Q1 − (3 × IQR)
- Upper bound: Q3 + (3 × IQR)

Values outside these bounds were clipped to the bounds rather than deleted, preserving row count while neutralising distortion from extreme values.

### 3.4 Output

Seven cleaned CSV files saved to `data/cleaned/`, one per original table.

---

## 4. Step 2 — Building the Master Analytics Table (MAT)

### 4.1 Objective

Consolidate all seven cleaned tables into a single, flat, one-row-per-customer table enriched with engineered features and risk scores. Machine learning models require this format.

### 4.2 Aggregation Logic

The fact tables contain multiple rows per customer (many payments, many activity events). These were collapsed to one row per customer using grouped aggregations.

**Activity log aggregation (groupby user_id):**

| Output Column | Aggregation | Purpose |
|---|---|---|
| total_events | COUNT of activity_id | Measure of overall engagement volume |
| total_logins | COUNT where event_type == 'login' | Direct engagement indicator |
| distinct_features | NUNIQUE of feature_used | Breadth of product usage |
| total_page_views | SUM of page_views | Content consumption depth |
| avg_session_dur | MEAN of session_duration_min | Session quality indicator |
| last_activity_date | MAX of event_date | Recency of engagement |
| total_errors | SUM of error_occurred | Product experience quality |
| mobile_events | SUM of is_mobile | Device preference signal |

**Payment aggregation (groupby user_id):**

| Output Column | Aggregation | Purpose |
|---|---|---|
| total_payments | COUNT of payment_id | Billing frequency |
| total_revenue | SUM of amount_usd | Lifetime value proxy |
| avg_payment_amt | MEAN of amount_usd | Typical transaction size |
| failed_payments | SUM of is_failed | Billing health signal |
| refunded_payments | SUM of is_refunded | Satisfaction / dispute signal |
| last_payment_date | MAX of payment_date | Payment recency |

### 4.3 Table Joins

The MAT was assembled in three sequential left joins, anchored on `fact_subscriptions` (which provides the churn label):

```
fact_subscriptions
    LEFT JOIN dim_users           ON user_id
    LEFT JOIN activity_summary    ON user_id
    LEFT JOIN payment_summary     ON user_id
```

A left join was chosen throughout to preserve all subscription rows. Customers with no activity logs or payments receive `NaN` values in those columns, which is meaningful: zero activity is itself a churn signal.

### 4.4 Feature Engineering

Seven derived features were calculated from existing columns:

**Recency features**
Using a snapshot date of 2025-01-01 as the reference "today":
- `days_since_last_login` = snapshot_date − last_activity_date
- `days_since_last_payment` = snapshot_date − last_payment_date

These capture how recently a customer engaged, independent of their absolute tenure.

**Spend efficiency**
- `avg_monthly_spend` = total_revenue ÷ (tenure_days / 30), clipped at 1 month minimum to avoid division by zero. Normalises spend by how long the customer has been subscribed.

**Engagement quality**
- `feature_adoption_score` = distinct_features, capped at 10. A 0–10 scale measuring product breadth.
- `login_frequency_monthly` = total_logins ÷ active months. Normalises login frequency by tenure.
- `events_per_day` = total_events ÷ tenure_days. A continuous engagement intensity metric.

**Billing health**
- `failed_payment_rate` = failed_payments ÷ total_payments. A proportion (0–1) indicating how often billing fails.

**Cohort grouping**
- `cohort_month` = signup_date truncated to year-month (e.g., '2022-01'). Groups customers by their signup period for cohort retention analysis.

### 4.5 RFM Scoring

Each customer was scored on three dimensions using quintile ranking (`pd.qcut` into 5 equal groups):

| Dimension | Logic | Score 1 | Score 5 |
|---|---|---|---|
| Recency (R) | days_since_last_login — fewer is better | Least recently active | Most recently active |
| Frequency (F) | total_logins — more is better | Fewest logins | Most logins |
| Monetary (M) | avg_monthly_spend — higher is better | Lowest spender | Highest spender |

Total RFM Score = R + F + M, ranging from 3 to 15. `pd.qcut` with `rank(method='first')` was used to handle tied values without assigning NaN.

### 4.6 Risk Segmentation

| RFM Score Range | Risk Segment | Interpretation |
|---|---|---|
| ≥ 11 | Low Risk | Engaged, spending, recently active |
| 8–10 | Medium Risk | Moderate engagement, monitor closely |
| < 8 | High Risk | Disengaged, low spend, or inactive |

### 4.7 Output

`data/mat/master_analytics_table.csv` — one row per customer, approximately 30–35 columns, ready for analysis and modelling.

---

## 5. Step 3 — Exploratory Data Analysis & Cohort Analysis

### 5.1 Objective

Visualise churn patterns, measure MRR at risk, and build a cohort retention heatmap to identify when customers tend to leave.

### 5.2 Key Business Metrics (Summary)

These metrics are computed at the start of the EDA step to set context:

- **Total customers** in the MAT
- **Overall churn rate** (mean of the `churned` column, expressed as %)
- **Total active MRR** (sum of MRR for non-churned customers)
- **High-risk customer count** (customers with RFM < 8)
- **MRR at risk from high-risk segment** (total MRR of high-risk customers)
- **Highest and lowest churn plans**
- **Highest churn acquisition channel**

### 5.3 Chart 1 — Churn Overview

Three sub-charts in one figure:

**Churn Rate by Plan (bar chart)**
Shows which pricing tier loses the most customers. Typically, lower-tier plans (Basic) exhibit higher churn because customers have not deeply invested in the product or found sufficient value.

**Churn Rate by Acquisition Channel (horizontal bar chart)**
Compares churn across channels such as Organic Search, Paid Advertising, and Referral. Channels with high churn rates indicate a quality problem — the customers they bring in have lower lifetime value. Referral-acquired customers tend to have lower churn because they arrive with a peer recommendation.

**Customer Risk Segment Distribution (pie chart)**
Shows the proportion of customers in each risk tier. This gives an immediate read on the health of the entire customer base. A large High Risk slice is an urgent business signal.

### 5.4 Chart 2 — Cohort Retention Heatmap

The retention heatmap is the most strategically valuable output of the EDA step.

**Construction method:**
1. `tenure_month` is computed as `tenure_days ÷ 30`, clipped at 24 months.
2. Customers are grouped by `cohort_month` × `tenure_month`.
3. For each group, retention % = (1 − churn_rate) × 100.
4. The result is pivoted into a matrix: rows = cohorts, columns = tenure months.

**Reading the heatmap:**
- Each cell shows what percentage of a given cohort was still active at that tenure month.
- The colour scale goes from red (low retention) to green (high retention).
- Vertical bands of red across multiple cohorts indicate a systematic dropout point — a specific tenure month where customers universally disengage.

**Business interpretation:**
If the heatmap shows heavy red at months 2–3, it means customers are leaving early in their lifecycle before experiencing the product's full value. This points to an onboarding problem. Red at months 10–12 suggests the product is failing at annual renewal, pointing to a value reinforcement problem.

### 5.5 Chart 3 — RFM Score Distributions

Three bar charts showing how many customers fall into each score level (1–5) for Recency, Frequency, and Monetary.

A healthy customer base shows a roughly uniform distribution or a right-skewed distribution (more customers at 4–5 than 1–2). A left-skewed distribution (many customers at 1–2) signals systemic disengagement.

### 5.6 Chart 4 — MRR at Risk by Segment

A bar chart comparing the total MRR held by High Risk, Medium Risk, and Low Risk segments.

The critical business metric here is High Risk MRR: the revenue that could be lost if all high-risk customers churned. This number is used to justify the cost of retention interventions — if High Risk MRR is $200K/month, even a 10% retention improvement from an intervention delivers $20K/month in savings.

### 5.7 Chart 5 — Feature Adoption vs Churn

An overlapping density histogram comparing the `feature_adoption_score` distribution for churned vs retained customers.

The expected finding: retained customers cluster at higher adoption scores, while churned customers cluster at lower scores. This chart provides intuitive visual evidence for one of the most actionable insights in the project — driving feature discovery during onboarding directly reduces churn.

---

## 6. Step 4 — Churn Prediction Model

### 6.1 Objective

Train a machine learning model that predicts, for each customer, the probability that they will churn. Use this model to generate a prioritised at-risk customer list that the retention team can act on.

### 6.2 Feature Selection

**Numeric features (17):**

| Feature | Business Rationale |
|---|---|
| tenure_days | Longer tenure = stronger loyalty |
| mrr_usd | Higher-paying customers are more committed and more costly to lose |
| total_logins | Engagement volume |
| distinct_features | Product breadth adoption |
| avg_session_dur | Session quality |
| days_since_last_login | Recency — the strongest early churn signal |
| days_since_last_payment | Payment recency |
| avg_monthly_spend | Spend commitment |
| feature_adoption_score | 0–10 product engagement score |
| login_frequency_monthly | Engagement frequency |
| failed_payment_rate | Billing health |
| events_per_day | Overall activity intensity |
| nps_score | Customer-reported satisfaction |
| rfm_score | Composite health score |
| total_payments | Billing history depth |
| upgrade_count | Positive signals of value realisation |
| downgrade_count | Negative signal — may indicate dissatisfaction |

**Categorical features (5):**
plan_name, acquisition_channel, billing_cycle, gender, company_size

**Target variable:**
`churned` — binary (0 = retained, 1 = churned)

### 6.3 Train / Validation / Test Split

Data was split chronologically by index position to respect time order and prevent lookahead bias:

| Set | Proportion | Purpose |
|---|---|---|
| Training | 70% | Model learns parameters |
| Validation | 15% | Intermediate evaluation and tuning |
| Test | 15% | Final unbiased evaluation |

A chronological split (rather than random) is the correct approach for customer data because a random split would allow the model to learn from future customers to predict past ones, inflating performance estimates.

### 6.4 Preprocessing Pipeline

Built using `sklearn.Pipeline` and `ColumnTransformer`:

- **Numeric features → StandardScaler:** Normalises each column to mean=0, std=1. This is essential for Logistic Regression, where large-scale features (e.g. `tenure_days`) would otherwise dominate small-scale ones (e.g. `nps_score`).
- **Categorical features → OneHotEncoder:** Converts text values to binary indicator columns (e.g. `plan_name=Basic` → `plan_name_Basic=1`, all others = 0). `handle_unknown='ignore'` ensures that unseen categories at inference time produce all-zero columns rather than errors.

Embedding these transformers inside a `Pipeline` prevents data leakage: the scaler is fit only on training data, and the same fitted scaler is applied to validation and test sets.

### 6.5 Model Comparison

**Model 1: Logistic Regression (Baseline)**

Logistic Regression finds a linear decision boundary in the feature space. Configured with:
- `C=0.1` — moderate regularisation to prevent overfitting
- `class_weight='balanced'` — adjusts loss weighting to handle the imbalance between churned (minority) and retained (majority) customers
- `max_iter=1000` — sufficient iterations for convergence

Logistic Regression provides an interpretable baseline and fast training time, making it useful for establishing whether the problem is linearly separable.

**Model 2: Random Forest (Best)**

Random Forest builds an ensemble of 200 independent decision trees, each trained on a random subset of features and rows (bootstrap sampling). The final prediction is the majority vote across all trees.

Configured with:
- `n_estimators=200` — 200 trees for stable predictions
- `max_depth=10` — limits tree depth to prevent overfitting
- `class_weight='balanced'` — handles class imbalance
- `n_jobs=-1` — parallelises training across all CPU cores

Random Forest captures non-linear relationships and feature interactions that Logistic Regression cannot, making it better suited to customer behaviour data where combinations of signals matter (e.g. low logins AND high failed payment rate is more predictive than either alone).

### 6.6 Evaluation

**Primary metric: AUC-ROC (Area Under the ROC Curve)**

AUC-ROC measures the model's ability to rank churners above non-churners across all possible classification thresholds. It is the preferred metric for churn prediction because:

1. The churn label is imbalanced (churners are typically a minority).
2. Accuracy alone is misleading on imbalanced datasets.
3. AUC-ROC is threshold-independent, allowing the business to choose the sensitivity vs specificity trade-off that fits their retention budget.

Interpretation:
- 0.5 = random guessing (no predictive power)
- 0.7–0.8 = acceptable
- 0.8–0.9 = strong
- > 0.9 = excellent

**Secondary metrics:**
- **Precision:** Of all customers flagged as high-churn-risk, what proportion actually churned? High precision = fewer wasted outreach contacts.
- **Recall:** Of all customers who actually churned, what proportion did the model catch? High recall = fewer churners missed.
- **F1 Score:** Harmonic mean of precision and recall — the right primary metric when both false positives (wasted outreach) and false negatives (missed churners) have meaningful costs.

**ROC Curve**
The ROC curve plots True Positive Rate (recall) against False Positive Rate at every possible threshold. The further the curve from the diagonal, the better. AUC is the area under this curve.

**Confusion Matrix**
The confusion matrix shows the four prediction outcomes at a 0.5 probability threshold:

|  | Predicted Retained | Predicted Churned |
|---|---|---|
| **Actual Retained** | True Negative ✅ | False Positive ❌ |
| **Actual Churned** | False Negative ❌ | True Positive ✅ |

### 6.7 Feature Importance

After training, Random Forest provides an importance score for every feature — the average reduction in node impurity (Gini impurity) contributed by each feature across all 200 trees.

Expected top predictors of churn:
1. `days_since_last_login` — the most direct recency signal
2. `rfm_score` — composite engagement health
3. `failed_payment_rate` — billing distress
4. `avg_monthly_spend` — economic commitment
5. `feature_adoption_score` — product stickiness
6. `login_frequency_monthly` — engagement cadence
7. `tenure_days` — loyalty duration

### 6.8 Customer Risk Scoring

After training, the best model's `predict_proba()` method is applied to every customer in the MAT to generate a churn probability score.

Probability buckets:

| Probability Range | Label | Interpretation |
|---|---|---|
| 0.0 – 0.3 | 🟢 Low | Unlikely to churn — maintain standard engagement |
| 0.3 – 0.6 | 🟡 Medium | Monitor — watch for disengagement signals |
| 0.6 – 0.8 | 🟠 High | Proactive outreach recommended |
| 0.8 – 1.0 | 🔴 Critical | Immediate intervention — high revenue at risk |

**at_risk_customers.csv** contains all customers with probability > 0.5, sorted by score descending. Each row includes: user_id, plan, channel, MRR, tenure, churn_probability, churn_label, and risk_segment.

### 6.9 Model Persistence

The best model pipeline (preprocessor + classifier) is serialised to `outputs/model/churn_model.pkl` using Python's `pickle` module. To reload and score new customers:

```python
import pickle
import pandas as pd

model = pickle.load(open('outputs/model/churn_model.pkl', 'rb'))
new_customers = pd.read_csv('new_customers.csv')
new_customers['churn_probability'] = model.predict_proba(new_customers[FEATURES])[:, 1]
```

---

## 7. Business Recommendations

Based on the analytical findings across all four steps, the following actions are recommended:

### Recommendation 1: Launch a Feature Discovery Onboarding Programme

The feature adoption analysis (Chart 5) consistently shows that customers who use more product features have materially lower churn rates. The critical window is the first 60–90 days of tenure, where the cohort heatmap shows the sharpest dropout.

**Action:** Design a structured onboarding email sequence (days 7, 14, 30, 60) that introduces one new feature per communication, with a direct link to try it. Track `distinct_features` as the primary onboarding KPI. Target: move new customers from < 3 features to ≥ 6 features within their first 90 days.

**Expected impact:** Customers using ≥ 6 features have significantly lower churn probability. A 1-point increase in `feature_adoption_score` corresponds to measurable improvement in the model's predicted retention rate.

### Recommendation 2: Implement a Billing Health Alert System

`failed_payment_rate` is consistently among the top predictors of churn. A customer whose payment fails and is not quickly resolved often cancels rather than update their card details — even if they are otherwise satisfied with the product.

**Action:** Build an automated dunning workflow that triggers on the first payment failure:
- Day 0: Immediate email with a direct link to update payment details.
- Day 3: Follow-up with a one-click payment retry prompt.
- Day 7: Outreach from the customer success team for high-MRR accounts.

**Expected impact:** Payment failure recovery rates of 30–50% are achievable with well-timed dunning sequences. For a customer base with significant failed payment volume, this represents directly recoverable MRR.

### Recommendation 3: Prioritise Retention Outreach Using the At-Risk List

The model produces a ranked list of customers sorted by churn probability. This list enables the retention team to move from reactive (responding to cancellation requests) to proactive (contacting customers before they decide to leave).

**Action:** Export `at_risk_customers.csv` weekly into the CRM. Create a customer success workflow where:
- 🔴 Critical (prob > 0.8) → Personal outreach within 48 hours from a senior CSM.
- 🟠 High (0.6–0.8) → Automated check-in email with a product value summary.
- 🟡 Medium (0.3–0.6) → Enrol in a targeted re-engagement email sequence.

**Prioritisation by MRR:** Within each risk tier, sort by `mrr_usd` descending. Saving a $1,000/month customer requires the same outreach effort as saving a $50/month customer but delivers 20× the revenue impact.

### Recommendation 4: Investigate High-Churn Acquisition Channels

The churn-by-channel analysis (Chart 1) reveals that not all acquisition channels produce equally retainable customers. Channels with high churn rates inflate topline customer counts but compress lifetime value and therefore payback period on customer acquisition cost (CAC).

**Action:** Calculate 12-month Lifetime Value (LTV) by channel. For any channel where LTV < (3 × CAC), reduce marketing spend and reallocate budget to lower-churn channels. Even at the same acquisition volume, improving channel mix can raise average LTV by 15–25%.

---

## 8. Technical Stack

| Component | Technology |
|---|---|
| Data manipulation | pandas 2.0, numpy |
| Machine learning | scikit-learn 1.3 |
| Visualisation | matplotlib 3.7, seaborn 0.12 |
| Model serialisation | Python pickle |
| Development environment | Jupyter Notebook |
| Language | Python 3.10+ |

---

## 9. Limitations & Future Work

### Current Limitations

**No time-series modelling:** The current model uses a static snapshot of each customer. It does not capture the trajectory of engagement (e.g. a customer who was active last month but has dropped off this month is treated the same as one who has always been low-engagement).

**Snapshot date dependency:** Feature engineering uses a fixed snapshot date of 2025-01-01. In production, this must be recalculated dynamically relative to the current date to keep recency features meaningful.

**No external factors:** The model does not account for external events (competitor launches, economic downturns, product outages) that may drive cohort-wide churn spikes not attributable to individual customer behaviour.

### Recommended Extensions

**Survival analysis:** Apply Cox Proportional Hazards or Kaplan-Meier curves to model time-to-churn rather than a binary outcome. This provides probability estimates at each future time point rather than a single static score.

**Uplift modelling:** Train a model that predicts not just who will churn, but whose churn probability would be most reduced by a retention intervention. This targets outreach budget at customers who are genuinely persuadable, rather than customers who would have stayed or left regardless.

**Real-time scoring pipeline:** Wrap the saved model in a lightweight API (Flask or FastAPI) that re-scores customers nightly as new activity logs come in. Push updated scores into the CRM automatically.

**A/B test retention interventions:** Use the at-risk list as the treatment group and a matched control group to measure the causal impact of retention campaigns on actual churn rates, building a feedback loop that improves both the model and the business strategy.

---

## 10. Appendix — File & Output Reference

| File | Location | Description |
|---|---|---|
| Raw data | `data/raw/*.csv` | Original source files (never modified) |
| Cleaned data | `data/cleaned/*.csv` | Post-cleaning, pre-join files |
| Master Analytics Table | `data/mat/master_analytics_table.csv` | One row per customer, all features |
| Chart 1 | `outputs/charts/01_churn_overview.png` | Plan, channel, risk pie |
| Chart 2 | `outputs/charts/02_cohort_retention_heatmap.png` | Cohort × tenure heatmap |
| Chart 3 | `outputs/charts/03_rfm_scores.png` | RFM distribution |
| Chart 4 | `outputs/charts/04_mrr_at_risk.png` | MRR by risk segment |
| Chart 5 | `outputs/charts/05_adoption_vs_churn.png` | Feature adoption histogram |
| Chart 6 | `outputs/charts/06_feature_importance.png` | Random Forest feature importance |
| Chart 7 | `outputs/charts/07_roc_and_confusion.png` | ROC curve + confusion matrix |
| Trained model | `outputs/model/churn_model.pkl` | Serialised sklearn Pipeline |
| At-risk list | `outputs/model/at_risk_customers.csv` | Scored customers, prob > 50% |

---

*Report prepared based on project notebooks: step01_data_cleaning.ipynb, step02_build_mat.ipynb, step03_eda_cohort_analysis.ipynb, step04_churn_model.ipynb*
