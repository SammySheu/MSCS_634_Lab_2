# MSCS 634 - Lab Assignment 2

**K-Nearest Neighbors (KNN) and Radius Neighbors (RNN) Classifier Analysis on the Wine Dataset**

## Purpose

The purpose of this lab is to explore the behavior of two neighbor-based classifiers **K-Nearest Neighbors (KNN)** and **Radius Neighbors (RNN)** using the Wine Dataset from scikit-learn. The lab investigates how tuning the `k` parameter (for KNN) and the `radius` parameter (for RNN) affects classification accuracy, visualizes the accuracy trends for each parameter, and compares the two models to determine when each is preferable.

## How to Run

```bash
# from the project directory
python3 -m venv .venv
source .venv/bin/activate
pip install scikit-learn pandas numpy matplotlib jupyter
jupyter notebook MSCS_634_Lab_2.ipynb
```

## Files

- `MSCS_634_Lab_2.ipynb` — Jupyter Notebook containing all code, outputs, plots, and analysis.
- `README.md` — This file.

## Dataset

- **Source:** `sklearn.datasets.load_wine`
- **Samples:** 178 wines
- **Features:** 13 chemical properties (alcohol, malic acid, ash, alcalinity of ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315, proline)
- **Classes:** 3 wine cultivars (class_0: 59 samples, class_1: 71 samples, class_2: 48 samples)
- **Split:** 80% training (142 samples) / 20% test (36 samples), stratified by class, `random_state=42`.

## Steps Performed

1. **Load and prepare** the Wine dataset; explore feature details and class distribution.
2. **KNN**: train and evaluate for `k ∈ {1, 5, 11, 15, 21}`.
3. **RNN**: train and evaluate for `radius ∈ {350, 400, 450, 500, 550, 600}` (with `outlier_label='most_frequent'` for safety).
4. **Visualize** accuracy trends for both classifiers and compare them side-by-side.

## Results

### KNN Accuracy

| k  | Accuracy |
|----|----------|
| 1  | 0.7778   |
| 5  | 0.8056   |
| 11 | 0.8056   |
| 15 | 0.8056   |
| 21 | 0.8056   |

### RNN Accuracy

| Radius | Accuracy |
|--------|----------|
| 350    | 0.7222   |
| 400    | 0.6944   |
| 450    | 0.6944   |
| 500    | 0.6944   |
| 550    | 0.6667   |
| 600    | 0.6667   |


## Challenges and Decisions

- One of the challenges I faced was understanding the spatial distribution of the wine dataset. Although AI can easily generate the code, what sets us apart is our understanding of the dataset and how to analyze it. I learned that proline values are much larger than those of the other features, so they can disproportionately affect the Euclidean distance. This dominance by a single feature can limit the model's maximum accuracy. Understanding this issue also shows why data normalization is important. It can prevents features with larger scales from dominating the analysis.

