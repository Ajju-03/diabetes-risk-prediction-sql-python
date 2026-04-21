-- 1. Create and select database
CREATE DATABASE IF NOT EXISTS medical_db;
USE medical_db;

-- ============================================================
-- TABLE 1: diabetes_data
-- ============================================================
CREATE TABLE IF NOT EXISTS diabetes_data (
    id                      INT AUTO_INCREMENT PRIMARY KEY,
    pregnancies             INT            NOT NULL,
    glucose                 FLOAT          NULL,         
    blood_pressure          FLOAT          NULL,         
    skin_thickness          FLOAT          NULL,         
    insulin                 FLOAT          NULL,         
    bmi                     FLOAT          NULL,         
    diabetes_pedigree       FLOAT          NOT NULL,
    age                     INT            NOT NULL,
    outcome                 TINYINT(1)     NOT NULL,     -- 1 = Diabetic, 0 = Not Diabetic
    created_at              TIMESTAMP      DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- TABLE 2: predictions
-- Stores every prediction the FastAPI model has made.
-- ============================================================
CREATE TABLE IF NOT EXISTS predictions (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    patient_id      INT            NULL,
    model_name      VARCHAR(100)   NOT NULL,
    model_version   VARCHAR(50)    NOT NULL,
    risk_score      FLOAT          NOT NULL,    -- probability 0.0 to 1.0
    prediction      TINYINT(1)     NOT NULL,    -- 1 = Diabetic, 0 = Not Diabetic
    risk_label      VARCHAR(50)    NOT NULL,    -- 'Low Risk' / 'Moderate Risk' / 'High Risk'
    predicted_at    TIMESTAMP      DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES diabetes_data(id)
        ON DELETE SET NULL
);

-- ============================================================
-- TABLE 3: experiment_log
-- ============================================================
CREATE TABLE IF NOT EXISTS experiment_log (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    mlflow_run_id   VARCHAR(255)   NOT NULL,
    model_name      VARCHAR(100)   NOT NULL,
    accuracy        FLOAT          NOT NULL,
    f1_score        FLOAT          NOT NULL,
    roc_auc         FLOAT          NOT NULL,
    logged_at       TIMESTAMP      DEFAULT CURRENT_TIMESTAMP
);

-- Quick check
SELECT 'Database setup complete.' AS status;
SELECT TABLE_NAME FROM information_schema.TABLES
WHERE TABLE_SCHEMA = 'medical_db';
