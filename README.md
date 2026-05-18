# 😄 Emotion-Aware Text Classifier 😢

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE) [![Build](https://img.shields.io/badge/build-passing-brightgreen)]()

A **CPU-efficient**, **explainable**, and **production-ready** pipeline that classifies text into emotional categories using classic NLP and machine learning techniques — no GPUs needed! 🤖✨

---

## 📚 Table of Contents 

1. 🚀 Quick Start  
2. 📂 Repo Structure  
3. 🔍 Project Overview  
4. ⚙️ Installation  
5. 🎯 Usage  
6. 🛠️ Scripts & Modules  
7. 📊 Results & Visuals  
8. 🤝 Contributing  
9. 📄 License  
10. 📫 Contact

---

## 🚀 Quick Start

```bash
git clone https://github.com/m-umar-raza/EMOTION-AWARE-TEXT-CLASSIFIER.git
cd EMOTION-AWARE-TEXT-CLASSIFIER
pip install -r requirements.txt
python src/preprocess.py --input data/balanced_dataset.csv --output data/clean_data.csv
python src/train.py --data data/clean_data.csv --model_output models/
jupyter notebook notebooks/emotion_text_classifier.ipynb
```

---

## 📂 Repo Structure

```
EMOTION-AWARE-TEXT-CLASSIFIER/
├── 📁 data/
│   └── balanced_dataset.csv                 # 500k tweets & reviews
├── 📁 docs/
│   └── Emotion_Aware_Text_Classifier_Report.docx  # Full project write-up
├── 📁 notebooks/
│   └── emotion_text_classifier.ipynb        # EDA, feature engineering, model demos
├── 📁 src/
│   ├── preprocess.py   # Text cleaning & feature engineering
│   ├── train.py        # Model training, evaluation & saving
│   └── utils.py        # VADER sentiment utility functions
├── 📄 requirements.txt  # Python libraries
├── 📄 .gitignore        # Ignored files
├── 📄 LICENSE           # MIT License
└── 📄 README.md         # You are here!
```

---

## 🔍 Project Overview

Emotion-aware sentiment analysis tags text with emotions:

- 😃 **positive**  
- 😡 **negative**

Leveraging **TF-IDF n-grams**, **VADER lexicon scores**, and **8 classic ML models**, we balance accuracy, interpretability, and speed on a 500k-row dataset of tweets & reviews.

---

## ⚙️ Installation

1. **Clone repo**  
   ```bash
   git clone https://github.com/akamohid/EMOTION-AWARE-TEXT-CLASSIFIER.git
   cd EMOTION-AWARE-TEXT-CLASSIFIER
   ```

2. **Create virtual env** _(recommended)_  
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎯 Usage

### 1. Preprocess Data  
Clean & tokenize raw text, outputting `clean_data.csv`:

```bash
python src/preprocess.py   --input data/balanced_dataset.csv   --output data/clean_data.csv
```

### 2. Train & Evaluate  
Train models, print metrics, and save best model + TF-IDF vectorizer under `models/`:

```bash
python src/train.py   --data data/clean_data.csv   --model_output models/
```

### 3. Interactive Analysis  
Open the Jupyter notebook for:
- EDA (class balance, word clouds)  
- Feature importance plots  
- Side-by-side model comparison  

```bash
jupyter notebook notebooks/emotion_text_classifier.ipynb
```

---

## 🛠️ Scripts & Modules

| File               | Purpose                                                                 |
| ------------------ | ----------------------------------------------------------------------- |
| `src/preprocess.py`| Cleans text (regex, tokenization, stopwords), generates `clean_text`.   |
| `src/train.py`     | Vectorizes text, trains 8 ML models, reports metrics, saves artifacts. |
| `src/utils.py`     | Helper: computes VADER compound score for each text sample.            |

---

## 📊 Results & Visuals

After training on an 80/20 split, top results:

| Model              | Accuracy | Precision | Recall | F1 Score |
| ------------------ | -------- | --------- | ------ | -------- |
| 🥇 Linear SVM      | 89%      | 0.88      | 0.90   | 0.89     |
| 🥈 Multinomial NB  | 88%      | 0.87      | 0.88   | 0.88     |
| 🥉 Logistic Reg.   | 86%      | 0.85      | 0.86   | 0.86     |
| 🔥 Gradient Boost  | 85%      | 0.84      | 0.86   | 0.85     |
| 🌳 KNN / Decision Tree | 78%  | 0.76      | 0.79   | 0.77     |

> **Visualizations** in the notebook:  
> - Token length distributions 📈  
> - Top unigrams & bigrams per emotion 🌟  
> - Confusion matrices & ROC curves 🎯

---

## 🤝 Contributing

1. **Fork** the repo  
2. **Create** a feature branch (`git checkout -b feature/YourIdea`)  
3. **Commit** your changes (`git commit -m "Add feature"`)  
4. **Push** (`git push origin feature/YourIdea`)  
5. **Open** a Pull Request  

Please follow the [code style guide](https://www.python.org/dev/peps/pep-0008/) and include tests when possible.

---

## 🛠️ Git Commands

```bash
git init
git remote remove origin 2>$null
git remote add origin https://github.com/akamohid/EMOTION-AWARE-TEXT-CLASSIFIER.git
git add .
git commit -m "Initial commit: Full emotion-aware text classifier implementation"
git branch -M main
git push -u origin main
```

---

## 👥 Team Members

- Mohid Arshad — [GitHub](https://github.com/akamohid) | [LinkedIn](https://linkedin.com/in/mohid-arshad-347180235/)
- Mohammad Umar — [GitHub](https://github.com/m-umar-raza) | [LinkedIn](https://linkedin.com/in/mohammad-umar-1147a62a6/)
- Mohammad Hasnain [LinkedIn](https://www.linkedin.com/in/mohammad-hasnain-3670452a7/)
- Tahir Mehmood [LinkedIn](https://www.linkedin.com/in/tahir-mehmood-622a412a0/)

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more details.

---

## 📫 Contact

👤 **Mohammad Umar**  
- GitHub: [m-umar-raza](https://github.com/m-umar-raza)  
- LinkedIn: [Mohammad Umar](www.linkedin.com/in/mohammad-umar-1147a62a6)  
- Email: m.umar81323@gmail.com  

---

✨ *Happy coding* ✨
