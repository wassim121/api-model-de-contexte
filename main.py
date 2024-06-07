from flask import Flask, request, jsonify
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

vectorizer = None
model = None
responses_dict = {}


def initialize_files():
    if not os.path.exists('data.csv'):
        df_data = pd.DataFrame(columns=['texte', 'contexte'])
        df_data.to_csv('data.csv', index=False)
    if not os.path.exists('responses.csv'):
        df_responses = pd.DataFrame(columns=['contexte', 'réponse'])
        df_responses.to_csv('responses.csv', index=False)


def update_model():
    global vectorizer, model, responses_dict
    df = pd.read_csv('data.csv')

    if df.empty:
        print("Le fichier data.csv est vide. Le modèle n'a pas été mis à jour.")
        return

    X = df['texte']
    y = df['contexte']

    if len(df) < 2:
        print("Pas assez de données pour entraîner le modèle.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    df_responses = pd.read_csv('responses.csv')
    responses_dict = pd.Series(df_responses.réponse.values, index=df_responses.contexte).to_dict()
    print(f'Model accuracy: {model.score(X_test_tfidf, y_test)}')


@app.route('/update', methods=['POST'])
def update():
    data = request.get_json()
    text = data.get('text')
    context = data.get('context')
    response = data.get('response')

    if not text or not context or not response:
        return jsonify({'error': 'Invalid input'}), 400

    initialize_files()
    df_data = pd.read_csv('data.csv')
    df_responses = pd.read_csv('responses.csv')

    new_data_row = pd.DataFrame({'texte': [text], 'contexte': [context]})
    df_data = pd.concat([df_data, new_data_row], ignore_index=True)
    df_data.to_csv('data.csv', index=False)

    new_response_row = pd.DataFrame({'contexte': [context], 'réponse': [response]})
    df_responses = pd.concat([df_responses, new_response_row], ignore_index=True)
    df_responses.to_csv('responses.csv', index=False)

    update_model()

    return jsonify({'message': 'Data and response added successfully'})

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({'error': 'Invalid input'}), 400

    if vectorizer is None or model is None:
        return jsonify({'error': 'Le modèle n\'est pas encore entraîné.'}), 500

    text_tfidf = vectorizer.transform([text])
    predicted_context = model.predict(text_tfidf)[0]
    response = responses_dict.get(predicted_context, 'Je suis désolé, je ne comprends pas votre demande.')

    return jsonify({'response': response})

@app.route('/retrain', methods=['POST'])
def retrain():
    initialize_files()
    update_model()
    return jsonify({'message': 'Le modèle a été réinitialisé et réentraîné.'})

if __name__ == '__main__':
    initialize_files()
    update_model()
    app.run(debug=True)
