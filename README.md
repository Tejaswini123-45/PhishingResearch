# An Efficiency-Based Analysis of a Deep Learning Phishing Detector

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 1. Project Overview

This project provides a critical analysis of the deep learning model presented in the research paper "Intelligent Deep Learning Based Cybersecurity Phishing Email Detection and Classification" by R. Brindha et al. While the paper's complex GRU-based model achieves a high accuracy of 99.72%, this project investigates whether that complexity is necessary.

To do this, we implement a classic, lightweight **Naive Bayes** machine learning model to serve as an efficient baseline. The results demonstrate that this simpler model can achieve a highly comparable accuracy, raising important questions about the trade-off between model complexity and practical efficiency in cybersecurity applications.

---

## 2. Research Context and Objective

* **Original Paper:** "Intelligent Deep Learning Based Cybersecurity Phishing Email Detection and Classification" (ICSOA-DLPEC model).
* **Identified Research Gap:** The paper's primary limitation is the lack of a baseline comparison. It does not justify its high computational complexity against simpler, more traditional models.
* **Project Objective:** To fill this gap by building a simple Naive Bayes classifier and comparing its performance, arguing that **efficiency** is a critical form of "improvement."

---

## 3. Methodology and Results

### Methodology
The approach uses a Multinomial Naive Bayes classifier, a standard algorithm for text classification. The text from the emails is converted into numerical features using a TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer. The model is trained on 80% of the dataset and evaluated on the remaining 20%.

### Results
The simple Naive Bayes model achieved a final accuracy of **98.81%** on the unseen test data. This result is less than 1% lower than the complex deep learning model, proving that a significant reduction in complexity is possible with only a marginal trade-off in performance.

---

## 4. Setup and Usage

Follow these steps to replicate the experiment.

### Prerequisites
* Python 3.9 or higher

### Installation
1.  Clone this repository to your local machine.
2.  Install the required Python libraries using pip:
    ```bash
    pip install pandas scikit-learn
    ```

### Dataset
This project uses the "Phishing Email Dataset" available on Kaggle.

* **Link:** [https://www.kaggle.com/datasets/subhajournal/phishing-email-dataset](https://www.kaggle.com/datasets/subhajournal/phishing-email-dataset)
* **Instructions:** Download the `phishing_email.csv` file from the link above. For the script to run correctly, place this CSV file in the same root folder as the `phishing_detector.py` script.

### Running the Script
Once the dataset is in place, you can run the model training and evaluation script from your terminal:
```bash
python phishing_detector.py
