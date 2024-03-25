import os
import json
import pickle


def load_data(directory):
    all_data = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            data_count = len(data['data'])
            print(f"Processing {filename} with {data_count} dialogues...")
            for item in data['data']:
                utterances = " ".join([u['utterance'] for u in item['body']])
                topic = item['header']['dialogueInfo']['topic']
                all_data.append((utterances, topic))
    return all_data


train_data = load_data('Training')
with open('training_data.pickle', 'wb') as file:
    pickle.dump(train_data, file)

# Validation 데이터 처리 및 저장
validation_data = load_data('Validation')
with open('validation_data.pickle', 'wb') as file:
    pickle.dump(validation_data, file)
