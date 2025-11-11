"""
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Carga de los modelos y objetos previamente entrenados
loaded_model = joblib.load("./pkl_objects/model2.pkl")
loaded_stop = joblib.load("./pkl_objects/stopwords.pkl")
loaded_vec = joblib.load("./pkl_objects/vectorizer2.pkl")

app = Flask(__name__)

# Función para clasificar documentos
def classify(document):
    label = {0: 'negativo', 1: 'positivo'}
    X = loaded_vec.transform([document])
    y = loaded_model.predict(X)[0]
    proba = np.max(loaded_model.predict_proba(X))
    return label[y], proba

# Ruta para manejar comentarios en formato JSON
@app.route('/analyze', methods=['POST'])
def analyze():
    if request.is_json:
        try:
            # Extraer el comentario del JSON recibido
            data = request.get_json()
            review = data.get('comment', None)

            if not review:
                return jsonify({'error': 'Falta el campo "comment" en el JSON'}), 400
            
            # Clasificar el comentario
            prediction, probability = classify(review)
            return jsonify({
                'comment': review,
                'prediction': prediction,
                
            })
        except Exception as e:
            return jsonify({'error': f'Error procesando la solicitud: {str(e)}'}), 500
    else:
        return jsonify({'error': 'La solicitud debe ser en formato JSON'}), 400

if __name__ == '__main__':
    app.run(debug=True)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Cargar modelos
loaded_model = joblib.load("./pkl_objects/model2.pkl")
loaded_vec = joblib.load("./pkl_objects/vectorizer2.pkl")

app = Flask(__name__)
CORS(app)  # Permitir peticiones externas

def classify(document):
    label = {0: 'negativo', 1: 'positivo'}
    X = loaded_vec.transform([document])
    y = loaded_model.predict(X)[0]
    proba = np.max(loaded_model.predict_proba(X))
    return label[y], proba

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.is_json:
        data = request.get_json()
        review = data.get('comment')
        if not review:
            return jsonify({'error': 'Falta el campo "comment" en el JSON'}), 400
        prediction, probability = classify(review)
        return jsonify({'comment': review, 'prediction': prediction})
    else:
        return jsonify({'error': 'La solicitud debe ser en formato JSON'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
