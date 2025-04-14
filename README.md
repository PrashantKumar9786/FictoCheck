
## 📰 Fake News Detection Using Machine Learning

This project is a machine learning-based approach to detect fake news using natural language processing (NLP) techniques. The goal is to classify news articles as either **Real** or **Fake** by analyzing their content. The project is implemented in Python using a Jupyter Notebook.

---

### 📁 Dataset

The project uses two datasets:
- **Fake.csv**: Contains fake news articles.
- **True.csv**: Contains real news articles.

Each dataset includes the following features:
- `title`: The title of the article
- `text`: The full text of the article
- `subject`: Category or topic
- `date`: Date of publication

---

### 🔧 Technologies Used

- Python
- Pandas & NumPy
- Matplotlib & Seaborn (for data visualization)
- Scikit-learn (for machine learning models and metrics)

---

### 🧠 ML Models Implemented

The following models are trained and evaluated:
- Logistic Regression
- Decision Tree Classifier
- Gradient Boosting Classifier
- Random Forest Classifier

Performance is evaluated using:
- Accuracy Score
- Classification Report (Precision, Recall, F1-score)

---

### 📊 Preprocessing Steps

1. Combined fake and real datasets into one with a new label column.
2. Cleaned text using regex and string functions (removing punctuation, lowercase conversion, etc.).
3. Converted text into numerical features using TF-IDF Vectorizer.
4. Split dataset into training and testing sets.

---

### ✅ Results

- The best-performing models showed **high accuracy and good precision/recall** in identifying fake vs real news.
- Visualizations and evaluation metrics helped understand model performance.

---

### 🚀 How to Run

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `fake_news_detection.ipynb` in Jupyter Notebook.
4. Run the cells to see data loading, cleaning, training, and evaluation.

---

### 📌 Future Improvements

- Incorporate deep learning models like LSTM or BERT for better accuracy.
- Build a simple web interface to input news and get predictions.
- Use more extensive and recent datasets for training.

---

