-- ============================================================
--  analysis_queries.sql  — Exploratory SQL Analysis
-- ============================================================

USE medical_db;

-- ── 1. PREVIEW DATA ───────────────────────────────────────────────────────────
-- See the first 5 rows to confirm data loaded correctly.
SELECT * FROM diabetes_data LIMIT 5;

-- ── 2. TOTAL RECORD COUNT ─────────────────────────────────────────────────────
SELECT COUNT(*) AS total_records FROM diabetes_data;

-- ── 3. CLASS DISTRIBUTION ─────────────────────────────────────────────────────
-- How many patients are Diabetic (1) vs Non-Diabetic (0)?
SELECT
    outcome,
    CASE outcome WHEN 1 THEN 'Diabetic' ELSE 'Non-Diabetic' END AS label,
    COUNT(*) AS total,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM diabetes_data), 1) AS pct -- pct of 1 & 0
FROM diabetes_data
GROUP BY outcome;

-- ── 4. INVALID ZERO VALUE COUNTS ──────────────────────────────────────────────
SELECT
    SUM(glucose        = 0) AS zero_glucose,
    SUM(blood_pressure = 0) AS zero_blood_pressure,
    SUM(skin_thickness = 0) AS zero_skin_thickness,
    SUM(insulin        = 0) AS zero_insulin,
    SUM(bmi            = 0) AS zero_bmi
FROM diabetes_data;

-- ── 5. AVERAGE STATS BY OUTCOME ───────────────────────────────────────────────
SELECT
    outcome,
    CASE outcome WHEN 1 THEN 'Diabetic' ELSE 'Non-Diabetic' END AS label,
    ROUND(AVG(glucose), 1)          AS avg_glucose,
    ROUND(AVG(bmi), 1)              AS avg_bmi,
    ROUND(AVG(age), 1)              AS avg_age,
    ROUND(AVG(insulin), 1)          AS avg_insulin,
    ROUND(AVG(blood_pressure), 1)   AS avg_blood_pressure
FROM diabetes_data
GROUP BY outcome;

-- ── 6. AGE GROUP RISK ANALYSIS ────────────────────────────────────────────────
-- Older patients have higher diabetes risk.
SELECT
    CASE
        WHEN age < 30 THEN 'Young (< 30)'
        WHEN age BETWEEN 30 AND 50 THEN 'Middle (30-50)'
        ELSE                           'Senior (> 50)'
    END                            AS age_group,
    COUNT(*)                       AS total_patients,
    SUM(outcome)                   AS diabetic_cases,
    ROUND(SUM(outcome) * 100.0 / COUNT(*), 1) AS diabetic_pct
FROM diabetes_data
GROUP BY age_group
ORDER BY diabetic_pct DESC;

-- ── 7. GLUCOSE RISK BRACKET ───────────────────────────────────────────────────
SELECT
    CASE
        WHEN glucose < 100  THEN 'Normal (< 100)'
        WHEN glucose < 140  THEN 'Pre-Diabetic (100-139)'
        ELSE                     'Diabetic Range (>= 140)'
    END                    AS glucose_bracket,
    COUNT(*)               AS patients,
    SUM(outcome)           AS diabetic_cases,
    ROUND(SUM(outcome) * 100.0 / COUNT(*), 1) AS diabetic_pct
FROM diabetes_data
WHERE glucose > 0
GROUP BY glucose_bracket
ORDER BY diabetic_pct DESC;

-- ── 8. TOP 10 HIGH RISK PATIENTS ──────────────────────────────────────────────
-- High glucose + high BMI + older age = highest risk profile.
SELECT
    id, pregnancies, glucose, bmi, age,
    diabetes_pedigree, outcome
FROM diabetes_data
WHERE glucose >= 140
  AND bmi >= 30
  AND age > 40
ORDER BY glucose DESC
LIMIT 10;
