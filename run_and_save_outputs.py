#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk import tokenize
from wordcloud import WordCloud

# Create output directory
os.makedirs('output', exist_ok=True)

print("=" * 60)
print("FAKE NEWS DETECTION - FULL OUTPUT")
print("=" * 60)

# ========================================
# 1. LOAD DATA
# ========================================
print("\n[1] Loading datasets...")
fake = pd.read_csv("Fake\\Fake.csv")
true = pd.read_csv("True\\True.csv")
print(f"    Fake shape: {fake.shape}")
print(f"    True shape: {true.shape}")

# ========================================
# 2. DATA PREPARATION
# ========================================
print("\n[2] Preparing data...")
fake['target'] = 'fake'
true['target'] = 'true'

data = pd.concat([fake, true]).reset_index(drop=True)
data = shuffle(data)
data = data.reset_index(drop=True)

data.drop(["date"], axis=1, inplace=True)
data.drop(["title"], axis=1, inplace=True)

print(f"    Combined shape: {data.shape}")
print(f"    Class distribution:\n{data['target'].value_counts().to_string()}")

# ========================================
# 3. TEXT PREPROCESSING
# ========================================
print("\n[3] Preprocessing text...")

# Lowercase
data['text'] = data['text'].apply(lambda x: x.lower())

# Remove punctuation
def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

data['text'] = data['text'].apply(punctuation_removal)

# Remove stopwords
nltk.download('stopwords', quiet=True)
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

print("    Preprocessing complete!")
print(f"    Sample cleaned text: {data['text'].iloc[0][:100]}...")

# ========================================
# 4. EDA CHARTS
# ========================================
print("\n[4] Generating EDA charts...")

# Chart 1: Subject Distribution
plt.figure(figsize=(12, 6))
data.groupby(['subject'])['text'].count().plot(kind="bar", color='steelblue')
plt.title('Number of Articles per Subject', fontsize=14)
plt.xlabel('Subject')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('output/01_subject_distribution.png', dpi=150)
plt.close()
print("    [4.1] Subject distribution chart saved")

# Chart 2: Class Distribution
plt.figure(figsize=(8, 5))
data.groupby(['target'])['text'].count().plot(kind="bar", color=['#e74c3c', '#2ecc71'])
plt.title('Fake vs Real News Distribution', fontsize=14)
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('output/02_class_distribution.png', dpi=150)
plt.close()
print("    [4.2] Class distribution chart saved")

# Chart 3: Word Cloud - Fake News
fake_data = data[data["target"] == "fake"]
all_words = ' '.join([text for text in fake_data.text])
wordcloud_fake = WordCloud(width=800, height=500, max_font_size=110,
                           collocations=False, background_color='white').generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud - Fake News', fontsize=14)
plt.tight_layout()
plt.savefig('output/03_wordcloud_fake.png', dpi=150)
plt.close()
print("    [4.3] Fake news word cloud saved")

# Chart 4: Word Cloud - Real News
real_data = data[data["target"] == "true"]
all_words = ' '.join([text for text in real_data.text])
wordcloud_real = WordCloud(width=800, height=500, max_font_size=110,
                           collocations=False, background_color='white').generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud - Real News', fontsize=14)
plt.tight_layout()
plt.savefig('output/04_wordcloud_real.png', dpi=150)
plt.close()
print("    [4.4] Real news word cloud saved")

# Chart 5 & 6: Most Frequent Words
token_space = tokenize.WhitespaceTokenizer()

def counter(text_df, column_text, quantity, filename, title):
    all_words = ' '.join([text for text in text_df[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns="Frequency", n=quantity)
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df_frequency, x="Word", y="Frequency", color='steelblue')
    ax.set(ylabel="Count")
    plt.title(title, fontsize=14)
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.savefig(f'output/{filename}', dpi=150)
    plt.close()
    return df_frequency

freq_fake = counter(data[data["target"] == "fake"], "text", 20,
                    "05_top20_words_fake.png", "Top 20 Words in Fake News")
freq_real = counter(data[data["target"] == "true"], "text", 20,
                    "06_top20_words_real.png", "Top 20 Words in Real News")
print("    [4.5-4.6] Top words charts saved")

# ========================================
# 5. MODEL TRAINING
# ========================================
print("\n[5] Training Decision Tree model...")

X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data.target, test_size=0.2, random_state=42)

print(f"    Train size: {X_train.shape[0]}")
print(f"    Test size: {X_test.shape[0]}")

pipe = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', DecisionTreeClassifier(criterion='entropy',
                                      max_depth=20,
                                      splitter='best',
                                      random_state=42))
])

model = pipe.fit(X_train, y_train)

# ========================================
# 6. EVALUATION
# ========================================
print("\n[6] Evaluating model...")

prediction = model.predict(X_test)
accuracy = round(accuracy_score(y_test, prediction) * 100, 2)
print(f"    Accuracy: {accuracy}%")

# Classification Report
report = metrics.classification_report(y_test, prediction, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('output/07_classification_report.csv')
print("    Classification report saved")

# Confusion Matrix
cm = metrics.confusion_matrix(y_test, prediction)
cm_df = pd.DataFrame(cm, index=['Actual Fake', 'Actual Real'],
                     columns=['Predicted Fake', 'Predicted Real'])
cm_df.to_csv('output/08_confusion_matrix.csv')
print("    Confusion matrix saved")

# Chart 7: Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title(f'Confusion Matrix (Accuracy: {accuracy}%)', fontsize=14)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('output/09_confusion_matrix_plot.png', dpi=150)
plt.close()
print("    [4.7] Confusion matrix plot saved")

# ========================================
# 7. PREDICTIONS ON TEST DATA
# ========================================
print("\n[7] Generating predictions on test data...")

predictions_df = pd.DataFrame({
    'text': X_test.values,
    'actual': y_test.values,
    'predicted': prediction
})
predictions_df.to_csv('output/10_test_predictions.csv', index=False)
print(f"    {len(predictions_df)} predictions saved to 10_test_predictions.csv")

# ========================================
# 8. CUSTOM TEST PREDICTIONS
# ========================================
print("\n[8] Testing with custom news articles...")

test_articles = [
    "BREAKING: Shocking revelation about government conspiracy, you won't believe what happened next!!!",
    "The Federal Reserve announced a 0.25% interest rate increase following their scheduled meeting.",
    "EXPOSED: Celebrity caught in massive scandal, mainstream media hiding the truth from you!",
    "Scientists at MIT published a new study on climate change effects on coastal cities.",
    "URGENT: Miracle cure discovered, big pharma doesn't want you to know about this secret remedy!",
    "The president signed a new executive order on immigration policy today at the White House.",
    "You won't BELIEVE what this politician did! Share before it gets deleted!!!",
    "A new report by the World Health Organization shows global health improvements in vaccination rates."
]

custom_results = []
for article in test_articles:
    pred = model.predict([article])[0]
    proba = model.predict_proba([article])[0]
    confidence = round(max(proba) * 100, 2)
    custom_results.append({
        'article': article[:80] + '...' if len(article) > 80 else article,
        'prediction': pred,
        'confidence': f"{confidence}%"
    })
    print(f"    [{pred.upper():4s}] ({confidence}%) {article[:60]}...")

custom_df = pd.DataFrame(custom_results)
custom_df.to_csv('output/11_custom_predictions.csv', index=False)
print("    Custom predictions saved")

# ========================================
# 9. SUMMARY STATISTICS
# ========================================
print("\n[9] Saving summary statistics...")

summary = {
    'Metric': [
        'Total Articles', 'Fake Articles', 'Real Articles',
        'Train Size', 'Test Size', 'Model', 'Criterion',
        'Max Depth', 'Accuracy (%)', 'Precision (Fake)',
        'Recall (Fake)', 'F1-Score (Fake)', 'Precision (Real)',
        'Recall (Real)', 'F1-Score (Real)'
    ],
    'Value': [
        len(data), len(fake_data), len(real_data),
        X_train.shape[0], X_test.shape[0], 'Decision Tree',
        'entropy', 20, accuracy,
        round(report['fake']['precision'], 4),
        round(report['fake']['recall'], 4),
        round(report['fake']['f1-score'], 4),
        round(report['true']['precision'], 4),
        round(report['true']['recall'], 4),
        round(report['true']['f1-score'], 4)
    ]
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv('output/12_model_summary.csv', index=False)
print("    Summary saved")

# ========================================
# 10. TOP WORDS DATA
# ========================================
freq_fake.to_csv('output/13_top20_words_fake.csv', index=False)
freq_real.to_csv('output/14_top20_words_real.csv', index=False)
print("    Top words data saved")

# ========================================
# FINAL SUMMARY
# ========================================
print("\n" + "=" * 60)
print("ALL OUTPUTS SAVED IN 'output/' FOLDER:")
print("=" * 60)
print("""
Charts (PNG):
  01_subject_distribution.png
  02_class_distribution.png
  03_wordcloud_fake.png
  04_wordcloud_real.png
  05_top20_words_fake.png
  06_top20_words_real.png
  09_confusion_matrix_plot.png

Data (CSV):
  07_classification_report.csv
  08_confusion_matrix.csv
  10_test_predictions.csv (all test set predictions)
  11_custom_predictions.csv (8 custom test articles)
  12_model_summary.csv (complete model stats)
  13_top20_words_fake.csv
  14_top20_words_real.csv
""")
print(f"Model Accuracy: {accuracy}%")
print("=" * 60)
