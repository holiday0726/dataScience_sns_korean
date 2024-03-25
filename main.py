from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
import json
import pickle


def combine_json_files(folder_path):
    combined_data = {'data': []}  # 새로운 JSON 구조 초기화
    total_items = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):  # JSON 파일인지 확인
            file_path = os.path.join(folder_path, file_name)dnjz
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                combined_data['data'].extend(json_data['data'])  # 데이터 결합
                total_items += json_data['numberOfItems']  # 아이템 수 더하기

    combined_data['numberOfItems'] = total_items  # 전체 아이템 수 업데이트
    return combined_data


def save_data_with_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


# # 사용 예시
folder_path = 'Training'  # JSON 파일이 있는 폴더 경로
combined_data = combine_json_files(folder_path)
folder_path = 'Validation'  # JSON 파일이 있는 폴더 경로
combined_data = combine_json_files(folder_path)

pickle_file_path = 'combined_data.pkl'  # 결합된 데이터를 저장할 pickle 파일 경로
save_data_with_pickle(combined_data, pickle_file_path)


def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


# # 저장된 pickle 파일 경로
# pickle_file_path = 'combined_data.pkl'

# # pickle 파일 불러오기
# loaded_data = load_data_from_pickle(pickle_file_path)

# # 데이터 추출
# X = [item['body'][i]['utterance'] for item in loaded_data['data']
#      for i in range(len(item['body']))]
# y = [item['header']['dialogueInfo']['topic']
#      for item in loaded_data['data'] for i in range(len(item['body']))]

# # 데이터 분할
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)

# example_data = [
#     "오늘은 마음의 평화를 찾는 시간을 갖기로 했다.",
#     "영화를 보러 가는 건 어때요?",
#     "이번 주말에는 친구들과 함께 바베큐 파티를 열어보는 건 어떨까요?"
# ]

# # 벡터화
# vectorizer = TfidfVectorizer()
# X_train_vec = vectorizer.fit_transform(X_train)
# X_test_vec = vectorizer.transform(X_test)

# # 모델 학습
# clf = MultinomialNB()
# clf.fit(X_train_vec, y_train)

# # 예측
# y_pred = clf.predict(X_test_vec)

# # 평가
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # 저장할 모델
# model = None  # 모델을 초기화하거나 훈련된 모델을 할당합니다.

# # 저장할 파일 경로
# model_file_path = 'naive_bayes_model.pkl'

# # 모델 저장
# with open(model_file_path, 'wb') as file:
#     pickle.dump(model, file)

# print("모델이 성공적으로 저장되었습니다.")

# example_data = [
#     "오늘은 마음의 평화를 찾는 시간을 갖기로 했다.",
#     "영화를 보러 가는 건 어때요?",
#     "이번 주말에는 친구들과 함께 바베큐 파티를 열어보는 건 어떨까요?"
# ]

# # TF-IDF 벡터 변환기 초기화 및 적합
# vectorizer = TfidfVectorizer()
# vectorizer.fit(training_texts)  # 훈련 데이터로 적합

# # 저장된 모델 불러오기
# model_file_path = 'naive_bayes_model.pkl'
# with open(model_file_path, 'rb') as file:
#     model = pickle.load(file)

# # 예제 데이터를 TF-IDF 벡터로 변환
# example_vectors = vectorizer.transform(example_data)

# # 변환된 데이터를 모델에 입력하여 예측 수행
# predictions = model.predict(example_vectors)

# # 예측 결과 출력
# for text, prediction in zip(example_data, predictions):
#     print(f"Text: {text}")
#     print(f"Predicted label: {prediction}")
