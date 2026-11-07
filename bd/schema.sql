CREATE DATABASE VetClinic;
GO

USE VetClinic;
GO


-- ============================================
-- ВЕТЕРИНАРЫ
-- ============================================

CREATE TABLE vets (
    vet_id INT PRIMARY KEY,
    full_name NVARCHAR(150) NOT NULL,
    phone NVARCHAR(30) NULL
);
GO


-- ============================================
-- ЖИВОТНЫЕ
-- ============================================

CREATE TABLE animals (
    animal_id INT PRIMARY KEY,
    nickname NVARCHAR(100) NOT NULL,
    number NVARCHAR(10) NOT NULL UNIQUE,   -- текст, чтобы сохранить ведущие нули (001)
    breed NVARCHAR(100) NOT NULL,
    gender NVARCHAR(20) NOT NULL,
    birth_date DATE NOT NULL
);
GO


-- ============================================
-- ДИАГНОЗЫ
-- ============================================

CREATE TABLE diagnoses (
    diagnosis_id INT PRIMARY KEY,
    name NVARCHAR(150) NOT NULL,
    is_chronic BIT NOT NULL,
    disease_group NVARCHAR(100) NOT NULL   -- группа болезни (для прогноза риска по группам)
);
GO


-- ============================================
-- ВЕТЕРИНАРНЫЕ ОСМОТРЫ
-- ============================================

CREATE TABLE examinations (
    examination_id INT PRIMARY KEY,

    animal_id INT NOT NULL,
    vet_id INT NOT NULL,
    diagnosis_id INT NOT NULL,

    exam_date DATE NOT NULL,

    age INT NOT NULL,
    lactation_number INT NOT NULL,
    milk_yield INT NOT NULL,
    lameness_score INT NOT NULL,

    previous_examinations_count INT NOT NULL,
    diseases_last_6_months INT NOT NULL,
    has_chronic_disease BIT NOT NULL,

    treatment NVARCHAR(500) NOT NULL,
    is_sick BIT NOT NULL,

    CONSTRAINT FK_examinations_animals
        FOREIGN KEY (animal_id)
        REFERENCES animals(animal_id),

    CONSTRAINT FK_examinations_vets
        FOREIGN KEY (vet_id)
        REFERENCES vets(vet_id),

    CONSTRAINT FK_examinations_diagnoses
        FOREIGN KEY (diagnosis_id)
        REFERENCES diagnoses(diagnosis_id)
);
GO
