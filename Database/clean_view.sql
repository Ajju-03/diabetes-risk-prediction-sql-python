-- ============================================================
--  clean_view.sql  — SQL VIEW: Replace invalid zeros with NULL
-- ============================================================

USE medical_db;

-- Drop and recreate so it's always in sync with the table schema
DROP VIEW IF EXISTS clean_diabetes_data;

CREATE VIEW clean_diabetes_data AS
SELECT
    id,
    pregnancies,
    NULLIF(glucose,        0) AS glucose,          
    NULLIF(blood_pressure, 0) AS blood_pressure,   
    NULLIF(skin_thickness, 0) AS skin_thickness,  
    NULLIF(insulin,        0) AS insulin,          
    NULLIF(bmi,            0) AS bmi,             
    diabetes_pedigree,
    age,
    outcome
FROM diabetes_data;

-- Verify the view works
SELECT 'View created successfully.' AS status;

-- Spot-check: count NULLs (should be > 0 now)
SELECT
    COUNT(*) - COUNT(glucose)        AS null_glucose,
    COUNT(*) - COUNT(blood_pressure) AS null_blood_pressure,
    COUNT(*) - COUNT(insulin)        AS null_insulin,
    COUNT(*) - COUNT(bmi)            AS null_bmi
FROM clean_diabetes_data;

-- Preview clean data
SELECT * FROM clean_diabetes_data LIMIT 5;
