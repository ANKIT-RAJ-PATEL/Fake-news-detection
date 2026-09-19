# Fake News Detection

A machine learning project that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP) techniques.

## Overview

Fake news is a growing problem in today's digital world. This project uses text classification to detect whether a given news article is fake or real based on its content.

## Technologies Used

- **Python** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning models and tools
- **NLTK** - Natural language processing (stopwords, tokenization)
- **WordCloud** - Visual representation of frequent words

## Project Workflow

1. **Data Loading** - Load fake and real news datasets (CSV files)
2. **Data Preprocessing**
   - Combine fake and real datasets with target labels
   - Shuffle the combined data
   - Remove unnecessary columns (date, title)
   - Convert text to lowercase
   - Remove punctuation
   - Remove stopwords
3. **Exploratory Data Analysis**
   - Articles per subject
   - Fake vs Real article distribution
   - Word clouds for fake and real news
   - Most frequent words analysis
4. **Model Training**
   - Text vectorization using CountVectorizer
   - TF-IDF transformation
   - Decision Tree Classifier with entropy criterion
5. **Evaluation**
   - Accuracy score
   - Confusion matrix visualization

## Dataset

The project uses two datasets:

| Dataset | Description |
|---------|-------------|
| `Fake.csv` | Collection of fake news articles |
| `True.csv` | Collection of real news articles |

Each dataset contains the following columns:
- `title` - Headline of the news article
- `text` - Full body text of the article
- `subject` - Topic category
- `date` - Publication date

## Installation

```bash
# Clone the repository
git clone https://github.com/ANKIT-RAJ-PATEL/Fake-news-detection.git

# Navigate to project directory
cd Fake-news-detection

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```

## Usage

```bash
python "fake news detection1.py"
```

## Results

The model uses a **Decision Tree Classifier** with the following configuration:
- **Criterion:** Entropy
- **Max Depth:** 20
- **Splitter:** Best
- **Test Split:** 20%

The project generates:
- Distribution plots of fake vs real articles
- Word clouds highlighting common terms in fake and real news
- Most frequent words bar charts
- Confusion matrix for model evaluation

## Author

**Ankit Raj Patel**

## License

This project is open source and available for educational purposes.
