# Sports vs Politics Text Classification

**Author**: B23CM1044
**Course**: CSL 7640 - Natural Language Understanding
**Assignment**: Problem 4 - Text Classification

## Project Overview

This project implements a binary text classifier to distinguish between **Sports** and **Politics** articles using various machine learning techniques and feature extraction methods.

## Dataset

- **Total Samples**: 97
- **Training Set**: 77 samples
- **Test Set**: 20 samples
- **Classes**: Sports, Politics (balanced distribution)
- **Source**: Synthetic data generated with realistic article structures

## Feature Extraction Methods

1. **Bag of Words (BoW)**
   - Simple word frequency counting
   - Features: 1000

2. **TF-IDF (Term Frequency-Inverse Document Frequency)**
   - Weighted word importance
   - Features: 1000

3. **N-grams
   - Bigrams (1-2 grams): 1000 features
   - Captures word sequences and context

## Machine Learning Models

### Models Tested:
1. **Naive Bayes** (Multinomial)
2. **Logistic Regression
3. **Support Vector Machine** (Linear kernel)
4. **Random Forest** (100 estimators)

## Results

### Best Performing Model
- **Model**: NaiveBayes_BoW
- **Test Accuracy**: 1.0000
- **F1-Score**: 1.0000
- **Cross-Validation**: 0.9733 (+/- 0.0533)

### Complete Results

| Model | Features | Accuracy | F1-Score | CV Mean |
|-------|----------|----------|----------|---------|
| NaiveBayes | BoW | 1.0000 | 1.0000 | 0.9733 |
| SVM | TF-IDF | 1.0000 | 1.0000 | 0.9867 |
| LogisticRegression | TF-IDF | 1.0000 | 1.0000 | 0.9867 |
| NaiveBayes | TF-IDF | 1.0000 | 1.0000 | 1.0000 |
| SVM | Bigram | 1.0000 | 1.0000 | 0.9733 |
| LogisticRegression | Bigram | 1.0000 | 1.0000 | 0.9867 |
| NaiveBayes | Bigram | 1.0000 | 1.0000 | 1.0000 |
| LogisticRegression | BoW | 0.9500 | 0.9499 | 0.9733 |
| SVM | BoW | 0.9500 | 0.9499 | 0.9733 |
| RandomForest | BoW | 0.9000 | 0.8990 | 0.9733 |

## Project Structure
```
.
├── B23CM1044_sports_politics_classifier.ipynb  # Main notebook
├── sports_politics_dataset.csv                 # Dataset
├── model_comparison_results.csv                # Results table
├── report_statistics.json                      # Detailed statistics
├── project_summary.txt                         # Summary report
├── model_*.pkl                                 # Saved models
├── vectorizer_*.pkl                            # Saved vectorizers
├── *.png                                       # Visualizations
└── README.md                                   # This file
```

## How to Use

### Training
```python
# Load and train models (see notebook for complete code)
# All models are saved automatically
```

### Prediction
```python
# Load the best model
import pickle

with open('model_LogisticRegression_TF-IDF.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer_tfidf.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Make prediction
text = "The basketball team won the championship"
text_vectorized = vectorizer.transform([text])
prediction = model.predict(text_vectorized)[0]
print(f"Prediction: {prediction}")
```

## Key Findings

1. **Feature Importance**: TF-IDF features generally outperformed simple BoW
2. **Model Performance**: Logistic Regression and SVM showed similar strong performance
3. **Robustness**: Cross-validation scores were consistent with test accuracy
4. **Limitations**:
   - Synthetic data may not capture all real-world variations
   - Model may struggle with articles that mix both topics
   - Limited to binary classification (sports vs politics only)


## Requirements
```
python>=3.7
scikit-learn>=0.24.0
pandas>=1.2.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## Installation
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

## License

This project is for educational purposes as part of CSL 7640 coursework.

## Contact

For questions or feedback, please contact B23CM1044.

---
*Generated as part of NLU Assignment 4*
