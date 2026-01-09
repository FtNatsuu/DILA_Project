# Entity Matching Pipeline – DILA Project

This project implements an end-to-end **Entity Matching / Entity Resolution pipeline** for identifying matching scientific publications across two bibliographic datasets: **DBLP** and **Google Scholar**.  
The project is developed as part of the course **Data Integration and Large-Scale Analysis**.

---

## Requirements

- Python **3.9+**
- The following Python libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `imbalanced-learn`
  - `jellyfish`
## Setup Instructions

### 1. Create and activate a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\activate
```
Linux / macOS:
```powershell
python3 -m venv venv
source venv/bin/activate
```
### 2. Install dependencies

```powershell
pip install pandas numpy scikit-learn imbalanced-learn jellyfish
```
### 3. Create output directories
```powershell
mkdir scores
mkdir eval
```
These directories are required because the scripts save result files into them and do not create them automatically.

## Running Instructions
### Task 01: Entity Matching
Run:
```powershell
python task1.py
```
Outputs are saved in the scores/ directory.
### Task 02: Feature Vector and ML Model
Run:
```powershell
python task2.py
```
Outputs are saved in the eval/ directory.
