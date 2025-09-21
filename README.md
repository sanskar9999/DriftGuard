# DriftGuard: Anesthetic Depth Early Warning System

[![License: AGPL v3 NC](https://img.shields.io/badge/License-AGPL%20v3--NC-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Docs License: CC BY-NC-SA 4.0](https://img.shields.io/badge/Docs%20License-CC%20BY--NC--SA%204.0-green.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Status: Proof of Concept](https://img.shields.io/badge/Status-Proof%20of%20Concept-orange.svg)](#)

DriftGuard is a research project aimed at developing a machine learning model to predict changes in a patient's anesthetic depth during surgery. By analyzing real-time, non-invasive vital signs, the system's goal is to provide an early warning to anesthesiologists, helping them prevent the patient from becoming too deeply anesthetized or waking up prematurely.
<img width="1886" height="999" alt="image" src="https://github.com/user-attachments/assets/78848b3c-7e30-4b40-b50f-d47bc3e95099" />

## The Problem: The Anesthetic Balancing Act

Maintaining the correct depth of anesthesia is a critical challenge. A patient who is "too light" risks waking up, while a patient who is "too deep" can suffer from prolonged recovery and other complications. The Bispectral Index (BIS) monitor helps by providing a score for the brain's state, but it is not always available.

**Our mission is to create a "virtual BIS" early warning system using only the vital signs that are universally available in every operating room.**

## The Dataset: VitalDB

This project is built on the [VitalDB Open Dataset](https://physionet.org/content/vitaldb/1.0.0/), a comprehensive collection of high-resolution data from 6,388 surgical cases. The dataset's richness is also its greatest challenge: it is extremely noisy, heterogeneous (the "Swiss cheese" problem), and contains numerous data artifacts.

## Our Journey: From Flawed Models to a Data-Centric Approach

This repository documents a rigorous, and often frustrating, journey of model development. Our initial attempts, which focused on complex architectures like LSTMs and Transformers on raw data, resulted in a series of conclusive failures.

We systematically proved that:
1.  The raw vital signs have almost **no direct correlation** with the BIS score.
2.  The dataset is too **heterogeneous and noisy** for models to learn a generalizable pattern without careful feature engineering.
3.  A **model-centric approach is doomed to fail**. No algorithm, no matter how complex, can fix a fundamental data problem.

Our key insight was to pivot to a **data-centric approach**. The solution was not a better model, but better features.

## Key Finding: The Drug is the Signal, Vitals are the Context

Our most important discovery came from visualizing the raw data. The primary driver of the BIS score is the administration of anesthetic drugs (like Propofol). The other vital signs (Heart Rate, Blood Pressure) are the context—they show how the patient's body is responding to the combined stress of the surgery and the drugs.

<img width="1790" height="790" alt="image" src="https://github.com/user-attachments/assets/89b178df-0766-4c59-a9a8-6a591053b66f" />

*Visualization for Patient 4755, clearly showing the relationship between Propofol infusion (orange), the body's stress response (green/purple), and the resulting BIS score (blue).*

## The Final Approach

Our final, successful strategy is built on this insight:
1.  **Honest Preprocessing:** We use a robust pipeline that handles missing data with forward-filling (`ffill`) instead of dropping data or filling with nonsensical zeros.
2.  **Drug-Centric Feature Engineering:** We create features that model the interaction between the anesthetic drugs, the body's stress response, and the brain's state.
3.  **Robust Modeling:** We use a LightGBM model, trained incrementally on the clean, engineered features to handle the massive scale of the data without memory issues.

## Repository Structure

```
DriftGuard/
│
├── notebooks/
│   ├── 01_Data_Reconnaissance.ipynb      # Initial census and analysis of the dataset.
│   ├── 02_Feature_Engineering.ipynb      # The script to generate the drug-centric features.
│   ├── 03_Model_Training.ipynb           # The final, successful model training script.
│   └── archive/                          # Folder containing all our previous failed attempts (for learning).
│
├── src/
│   └── preprocessing.py                  # Python script with the core preprocessing functions.
│
├── visualizations/
│   └── patient_4755_timeline.png         # Key visualizations and plots.
│
├── .gitignore                            # Standard gitignore for Python projects.
├── LICENSE                               # MIT License file.
└── README.md                             # You are here.
```

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/DriftGuard.git
    cd DriftGuard
    ```
2.  **Set up the environment:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    (You will need to create a `requirements.txt` file with the necessary libraries).
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the notebooks:**
    Follow the notebooks in numerical order, starting with `01_Data_Reconnaissance.ipynb`.

## Current Status & Future Work

This project is currently a **successful proof-of-concept**. We have proven that a data-centric approach with intelligent feature engineering can find a real, predictive signal in this noisy dataset.

The next steps are to:
-   Expand the drug-centric feature set with more complex interaction terms.
-   Systematically tune the hyperparameters of the final LightGBM model.
-   Test the model's performance on a hold-out set of patients it has never seen.

## License

This project has a dual-license setup:

Code: Licensed under AGPL-3.0 (Non-Commercial).
-    Free to use, modify, and share for academic and research purposes only.
-    Commercial use (e.g., clinical deployment, startup products, SaaS) requires prior written permission.

Documentation & Research Materials: Licensed under CC BY-NC-SA 4.0.
-    Free to read, share, and build upon for non-commercial purposes, with attribution.
-    Derivatives must carry the same license.


For commercial licensing or collaborations, please contact: verma.sanskar@gmail.com maria.ahmed0009@gmail.com
