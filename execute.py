import pickle
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imblearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file.readlines()]
    return stopwords


file_path = 'stopwords.txt'

stopwords = load_stopwords(file_path)

with open('training_data.pickle', 'rb') as file:
    train_data = pickle.load(file)
print('load complete trainin data')

with open('validation_data.pickle', 'rb') as file:
    validation_data = pickle.load(file)
print('load validation trainin data')

X_train, y_train = zip(*train_data)
X_validation, y_validation = zip(*validation_data)

# X_train_sampled, _, y_train_sampled, _ = train_test_split(X_train, y_train, train_size=0.5, random_state=42)

model = make_pipeline_imblearn(TfidfVectorizer(stop_words=stopwords),
                               SMOTE(random_state=42),
                               LogisticRegression(max_iter=500,random_state=42, n_jobs=-1))
print('create pipeline')
model.fit(X_train, y_train)
# model.fit(X_train_sampled, y_train_sampled)
print('complete fit')

print('start predict')
y_pred = model.predict(X_validation)

print(classification_report(y_validation, y_pred))

with open('logistic_regression.pkl', 'wb') as f:
    pickle.dump(model, f)
