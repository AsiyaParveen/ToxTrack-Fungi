# ToxTrack-Fungi
ToxTrack Fungi: AI-Powered Fungal Diagnostic System ToxTrack Fungi is a high-precision Machine Learning application designed to classify mushroom specimens into Edible or Toxic categories. By analyzing 22 distinct biological markers, the system provides a safety profile with a balanced focus on accuracy and consumer safety (Recall).
Here is a professionally structured `README.md` content for your GitHub repository. It focuses on the technical architecture, the high-performance metrics you achieved, and the "Explainable AI" aspect of the project.

---

#  ToxTrack Fungi: AI-Powered Fungal Diagnostic System

**ToxTrack Fungi** is a high-precision Machine Learning application designed to classify mushroom specimens into **Edible** or **Toxic** categories. By analyzing 22 distinct biological markers, the system provides a safety profile with a balanced focus on accuracy and consumer safety (Recall).

## Model Architecture

The core engine utilizes a **Decision Tree Classifier** optimized via **Entropy (Information Gain)**. This model was chosen for its interpretability, allowing for "Explainable AI" where the path from specimen features to the final classification can be clearly visualized and audited.

### Technical Performance

The model was evaluated using an 80/20 train-test split, achieving near-perfect metrics:

| Metric | Value |
| --- | --- |
| **Accuracy** | **98.21%** |
| **Recall (Safety)** | **99.48%** |
| **Precision** | **96.84%** |
| **F1-Score** | **98.14%** |

### Confusion Matrix Analysis

The model's performance on the test set demonstrates its reliability, particularly in minimizing dangerous misclassifications:

* **True Positives (Edible):** 828
* **True Negatives (Poisonous):** 768
* **False Positives (Toxic marked Edible):** 4 (Critically low for safety)
* **False Negatives:** 25

---

##  Tech Stack & Implementation

* **Core Logic:** `Python 3.x`, `Scikit-Learn`
* **Data Manipulation:** `Pandas`, `NumPy`
* **Frontend UI:** `Streamlit` with custom CSS injection
* **Visualization:** `Matplotlib` / `Graphviz` for Decision Tree mapping

### Key Features

* **Dynamic UI Validation:** The interface dynamically generates input options based on the training dataset to prevent out-of-bounds errors.
* **Single-Card Navigation:** A seamless user flow that categorizes 22 features into 5 logical biological groups (Cap, Gills, Stalk, Veil, Ecology).
* **Entropy-Based Logic:** The system identifies that **Odor** is the most significant predictor of toxicity, followed by **Spore Print Color**.

---



2. **Install Dependencies:**
```bash
pip install streamlit pandas scikit-learn

```


3. **Run the App:**
```bash
streamlit run app.py

```



---
---

### 💡 Recommendation for your GitHub Profile

To make this project stand out even more, I recommend you add a **folder** in your repository named `visualizations/` and upload:

1. A screenshot of your **Confusion Matrix**.
2. An image of your **Decision Tree Graph**.
3. A **GIF** of you using the Streamlit app.

**Would you like me to help you write the code to generate a "Feature Importance" bar chart to add to this README?**
