def predict_with_model(model_and_vectorizer_path, text_data):
    # 모델 및 벡터화기 불러오기
    with open(model_and_vectorizer_path, 'rb') as file:
        model, vectorizer = pickle.load(file)
    
    text_vectors = vectorizer.transform(text_data)
    predictions = model.predict(text_vectors)

    return predictions
