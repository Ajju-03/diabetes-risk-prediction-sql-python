# create database for diabtes prediction
CREATE DATABASE medical_db;

# use database
USE medical_db;

# create table
CREATE TABLE diabetes_data (
    pregnancies INT,
    glucose INT,
    blood_pressure INT,
    skin_thickness INT,
    insulin INT,
    bmi FLOAT,
    diabetes_pedigree FLOAT,
    age INT,
    outcome INT 
);    

SELECT * FROM diabetes_data LIMIT 5;