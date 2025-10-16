# Prompt Injection Detection Using Machine Learning

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-complete-success)

A machine learning project for detecting prompt injection attacks in Large Language Model (LLM) systems using supervised classification algorithms.

> **📓 Main Deliverable**: [`notebooks/final_project_streamlined.ipynb`](notebooks/final_project_streamlined.ipynb) - Complete end-to-end analysis

---

## 📋 Problem Description

**Prompt injection** is a critical security vulnerability in LLM-powered applications where attackers craft malicious inputs to:
- Override system instructions and safety guidelines
- Leak sensitive information or internal prompts
- Execute unauthorized commands or code
- Manipulate model behavior for harmful purposes

As organizations deploy LLM chatbots and AI assistants for public use, detecting and preventing these attacks is essential to protect both users and systems from exploitation.

**Project Goal**: Build a binary classification system to automatically detect malicious prompts before they reach the LLM, using machine learning trained on labeled examples of prompt injection attacks.

---

## 📊 Dataset

**Malicious Prompt Detection Dataset (MPDD)**

- **Source**: [Kaggle - MPDD Dataset](https://www.kaggle.com/datasets/mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd)
- **Author**: Mohammed Amine Jebbar
- **License**: CC0 Public Domain
- **Size**: 39,234 samples
- **Task**: Binary classification (0 = Safe, 1 = Malicious)
- **Class Distribution**: Perfectly balanced (50% benign, 50% malicious)

### Features
- **Input**: Text prompts (variable length strings)
- **Target**: `isMalicious` (binary: 0 or 1)

**Citation**:
```
Jebbar, M. A. (2024). Malicious Prompt Detection Dataset (MPDD).
Kaggle. https://www.kaggle.com/datasets/mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd
```

---

## 🔧 Requirements & Installation

### Prerequisites

- **Python**: 3.8 or higher
- **pip**: Package manager

### Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
statsmodels>=0.13.0
nltk>=3.8.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/prompt-injection.git
cd prompt-injection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (automated in notebook):
```python
import nltk
nltk.download('punkt_tab')
```

---

## 🎯 Approach

### 1. Feature Engineering (19 Features)

Created 19 carefully selected features optimized for prompt injection detection:

**Basic Text Statistics (5 features)**:
- `char_count`, `word_count`, `avg_word_length`, `sentence_count`, `unique_word_ratio`

**Injection Pattern Detection (5 features)**:
- `sql_injection_keywords`: SQL commands (SELECT, DROP, DELETE, etc.)
- `command_injection_keywords`: Shell commands (bash, curl, sudo, etc.)
- `instruction_override_keywords`: Override phrases (ignore, disregard, forget, etc.)
- `jailbreak_phrases`: Jailbreak attempts (DAN, developer mode, etc.)
- `encoding_attempts`: Obfuscation indicators (base64, encode, hex, etc.)

**Code Injection Markers (4 features)**:
- `has_code_blocks`, `has_html_tags`, `has_javascript`, `has_shell_commands`

**Security Features (2 features)**:
- `has_credentials`: Password/token references
- `has_privilege_escalation`: Admin/root references

**Prompt Engineering Attacks (2 features)**:
- `role_playing_keywords`: Role-play attempts (act as, pretend, etc.)
- `system_prompt_leak`: System prompt extraction attempts

**Special Character Ratio (1 feature)**:
- `special_char_ratio`: Non-alphanumeric character density

### 2. Feature Selection via Cross-Validation

- Trained Random Forest classifier to rank features by importance
- Tested feature counts from 1 to 19 using 5-fold cross-validation
- Evaluated recall, precision, F1, and accuracy for each feature subset
- **Selected top 9 features** based on optimal cross-validation performance

**Top 9 Selected Features** (by importance):
1. `special_char_ratio`
2. `char_count`
3. `word_count`
4. `avg_word_length`
5. `unique_word_ratio`
6. `sentence_count`
7. `instruction_override_keywords`
8. `sql_injection_keywords`
9. `system_prompt_leak`

### 3. Models Tested

Trained and evaluated 4 classification algorithms:

1. **Logistic Regression** - Linear baseline model
2. **Random Forest** - Ensemble of decision trees
3. **Gradient Boosting** - Sequential boosting ensemble
4. **Support Vector Machine (SVM)** - RBF kernel for non-linear classification

### 4. Evaluation Metrics

- **Recall**: Percentage of malicious prompts correctly detected (minimize false negatives)
- **Precision**: Accuracy of malicious predictions (minimize false positives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (overall discriminative power)

---

## 🏆 Key Results

### Best Model: Gradient Boosting

| Metric | Score |
|--------|-------|
| **Recall** | **89.57%** |
| **Precision** | **97.31%** |
| **F1-Score** | **93.28%** |
| **ROC-AUC** | **96.18%** |

### Model Comparison

| Rank | Model | Recall | Precision | F1-Score | ROC-AUC |
|------|-------|--------|-----------|----------|---------|
| 1 | **Gradient Boosting** | 89.57% | 97.31% | 93.28% | 96.18% |
| 2 | Random Forest | 89.04% | 98.42% | 93.50% | 96.59% |
| 3 | SVM (RBF) | 88.68% | 97.72% | 92.98% | 95.48% |
| 4 | Logistic Regression | 81.62% | 89.07% | 85.18% | 92.09% |

### Key Insights

**Why Gradient Boosting Performed Best**:
- **Highest Recall (89.57%)**: Catches nearly 90% of malicious prompts - best detection rate among all models
- **Strong Precision (97.31%)**: Very few false alarms - only 2.69% of safe prompts incorrectly flagged
- **Excellent F1-Score (93.28%)**: Best balance between precision and recall
- **Sequential Learning**: Builds trees iteratively, correcting errors from previous trees for improved accuracy

**Security Implications**:
- **False Negative Rate: 10.43%** - About 10% of attacks slip through (lower than other models)
- **False Positive Rate: 2.69%** - Minimal disruption to legitimate users
- **Practical Trade-off**: Prioritizes recall (catching attacks) while maintaining acceptable precision
- **Best for Security**: Highest recall makes it optimal for security-critical applications where missing attacks is costlier than false alarms

**Model Blind Spot**:
- Missed attacks tend to have **shorter word lengths** and **lower feature complexity**
- Suggests attackers may evade detection with concise, simple malicious prompts
- Future work: Add features targeting terse injection techniques

---

## 🚀 How to Run

> **Main Notebook**: All analysis is contained in `notebooks/final_project_streamlined.ipynb`

### Option 1: Jupyter Notebook

1. **Start Jupyter**:
```bash
jupyter notebook
```

2. **Open the main notebook**:
   - Navigate to `notebooks/final_project_streamlined.ipynb`

3. **Run all cells**:
   - Click `Kernel > Restart & Run All`
   - Or run cells sequentially with `Shift + Enter`

### Option 2: JupyterLab

```bash
jupyter lab
```
Then open `notebooks/final_project_streamlined.ipynb` from the file browser

### Expected Runtime

- **Total execution time**: ~5-10 minutes
- **Most time-consuming steps**:
  - Feature engineering: ~1-2 minutes
  - Cross-validation (19 feature counts × 5 folds): ~3-5 minutes
  - Model training (4 models): ~1-2 minutes

---

## 🔮 Future Work

1. **Expand Feature Set**:
   - Add TF-IDF or word embeddings for semantic understanding
   - Include character n-grams for obfuscation detection
   - Extract patterns from successful attacks in production

2. **Deep Learning Approaches**:
   - Fine-tune transformer models (BERT, RoBERTa) for contextual detection
   - Compare performance vs computational cost trade-offs

3. **Adversarial Robustness**:
   - Generate adversarial prompts to test model resilience
   - Implement adversarial training to improve robustness
   - Develop confidence calibration for uncertain predictions

4. **Production Deployment**:
   - Build REST API for real-time prompt screening
   - Integrate with LLM gateways and prompt filters
   - Implement feedback loop for continuous model improvement

5. **Explainability**:
   - Add SHAP or LIME explanations for flagged prompts
   - Provide interpretable risk scores for security teams

---

## 📚 References & Attribution

### Dataset
```
Jebbar, M. A. (2024). Malicious Prompt Detection Dataset (MPDD).
Kaggle. https://www.kaggle.com/datasets/mohammedaminejebbar/malicious-prompt-detection-dataset-mpdd
License: CC0 Public Domain
```

### Tools & Libraries
- **scikit-learn**: Machine learning models and evaluation metrics
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **SciPy & statsmodels**: Statistical analysis
- **NLTK**: Natural language processing (sentence tokenization)

### Inspiration
- LLM security research and prompt injection attack taxonomies
- Machine learning for cybersecurity applications

---

**⭐ If you found this project useful, please consider giving it a star!**

**📧 Questions or suggestions? Feel free to open an issue or reach out.**
