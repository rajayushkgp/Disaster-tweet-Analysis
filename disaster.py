import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/content/train (1).csv')
df1 = pd.read_csv('/content/test.csv')
df.head()
x = df
sns.countplot(y=x.target);
counts = df['target'].value_counts()
total_count = counts.sum()
ratios = counts / total_count
print(ratios)
x.isnull().sum()
x.duplicated().sum()
data_types = df.dtypes
unique_counts = df.nunique()
numerical_vars = data_types[data_types != 'object'].index.tolist()
categorical_vars = data_types[data_types == 'object'].index.tolist()
print("Numerical Variables:", numerical_vars)
print("Categorical Variables:", categorical_vars)
df['keyword'].fillna('Unknown', inplace=True)
df['location'].fillna('Unknown', inplace=True)
plt.figure(figsize=(16,9))
sns.countplot(y=x.keyword, order =
x.keyword.value_counts().iloc[:50].index)
plt.title('Top 50 keywords')
plt.show()
w_d = x[x.target==1].keyword.value_counts().head(10)
w_nd = x[x.target==0].keyword.value_counts().head(10)
plt.figure(figsize=(20,5))
plt.subplot(121)
sns.barplot(w_d, color='c')
plt.title('Top keywords for disaster tweets')
plt.subplot(122)
sns.barplot(w_nd, color='y')
plt.title('Top keywords for non-disaster tweets')
plt.show()
total_count_disaster = w_d.sum()
total_count_non_disaster = w_nd.sum()
print("Total count of all values for disaster tweets:",
total_count_disaster)
print("Total count of all values for non-disaster tweets:",
total_count_non_disaster)
# For disaster tweets
print("Individual keyword value counts for disaster tweets:")
for keyword, count in w_d.items():
print(f"{keyword}: {count}")
# For non-disaster tweets
print("\nIndividual keyword value counts for non-disaster tweets:")
for keyword, count in w_nd.items():
print(f"{keyword}: {count}")
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def preprocess_text(text):
text = text.lower()
text = re.sub(r'[^a-zA-Z\s]', '', text)
tokens = word_tokenize(text)
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]
preprocessed_text = ' '.join(filtered_tokens)
return preprocessed_text
df['preprocessed_text'] = df['text'].apply(preprocess_text)
print(df.head())
from sklearn.feature_extraction.text import CountVectorizer,
TfidfVectorizer
corpus = df['text'].tolist()
count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(corpus)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
bow_df = pd.DataFrame(bow_matrix.toarray(),
columns=count_vectorizer.get_feature_names_out())
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
columns=tfidf_vectorizer.get_feature_names_out())
print("Bag-of-Words (BoW) Matrix:")
print(bow_df.head())
print("\nTF-IDF Matrix:")
print(tfidf_df.head())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
x = df['text']
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
random_state=42)
tfidf_vectorizer = TfidfVectorizer()
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lgbm_model = LGBMClassifier(n_estimators=100, random_state=42)
stacking_model = StackingClassifier(estimators=[('rf', rf_model), ('lgbm',
lgbm_model)], final_estimator=LGBMClassifier())
stacking_model.fit(x_train_tfidf, y_train)
y_pred_stacking = stacking_model.predict(x_test_tfidf)
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print("Accuracy of Stacking Classifier:", accuracy_stacking)
X_test_new = df1['text']
X_test_new_tfidf = tfidf_vectorizer.transform(X_test_new)
y_pred_test = stacking_model.predict(X_test_new_tfidf)
predictions_df = pd.DataFrame({'text_column': X_test_new,
'predicted_target': y_pred_test})
predictions_df.to_csv('/content/drive/MyDrive/collar.xlsx', index=False
