# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Data

df = pd.read_csv('phishing_email.csv')

# 2. Prepare the Data
# 'body' and 'label' are the column names in the CSV.
X = df['body']
y = df['label']

# This splits the data: 80% for training our model, 20% for testing it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Turn Text into Numbers
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Build and Train Our Simple Model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Test Our Model and See the Results
# We use the 20% of data we kept aside to see how well our model learned.
predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

# Print the results
print(f"Accuracy of our simple model: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))