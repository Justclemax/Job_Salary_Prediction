# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A machine learning project to predict job salaries from the [Job Salary Prediction Dataset](https://www.kaggle.com/datasets/nalisha/job-salary-prediction-dataset) (250,000 records, CC0 license). The target variable is `salary` (annual, integer).

## Environment Setup

Python 3.12 virtual environment at `env/`. Activate before running anything:

```bash
source env/bin/activate
```

Currently installed: `numpy`, `pandas`, `python-dateutil`. Install additional packages into this env.

## Dataset

**Source:** Kaggle — download `job_salary_prediction_dataset.csv` (2.999 MB zip) and place in `data/raw/`.

**Features:**
| Column | Type | Description |
|--------|------|-------------|
| `job_title` | Text | Job role (e.g., Data Analyst, AI Engineer) |
| `experience_years` | Integer | Years of professional experience |
| `education_level` | Text | Highest education level |
| `skills_count` | Integer | Number of technical skills |
| `industry` | Text | Industry sector |
| `company_size` | Text | small / medium / large |
| `location` | Text | Job location or region |
| `remote_work` | Text | Remote work availability |
| `certifications` | Integer | Number of professional certifications |
| `salary` | Integer | **Target variable** — annual salary |

## Data Directory Structure

```
data/
├── raw/       # Original dataset (job_salary_prediction_dataset.csv)
├── clean/     # Preprocessed/feature-engineered data
└── models/    # Serialized trained models
```