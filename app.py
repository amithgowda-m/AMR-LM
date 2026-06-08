from flask import Flask, request, jsonify, render_template
import logging
import traceback
from predictor import get_predictor
from database import save_prediction, get_history

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize predictor on startup
try:
    predictor = get_predictor()
    logger.info("Predictor initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    predictor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({"error": "Model failed to load. Please check server logs."}), 500

    try:
        data = request.get_json()
        if not data or 'sequence' not in data:
            return jsonify({"error": "No sequence provided in JSON payload. Key must be 'sequence'."}), 400

        sequence = data['sequence']
        
        # Simple validation
        if not all(c in 'ACGTUNacgtun\n\r ' for c in sequence):
            return jsonify({"error": "Invalid characters in sequence. Only A, C, G, T, U, N are allowed."}), 400

        # Perform prediction
        result = predictor.predict(sequence)
        
        if "error" in result:
            return jsonify(result), 400
            
        # Save to database
        db_id = save_prediction(
            sequence=sequence,
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"]
        )
        result["id"] = db_id
            
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error during prediction: {traceback.format_exc()}")
        return jsonify({"error": "An internal error occurred during prediction."}), 500

@app.route('/api/history', methods=['GET'])
def history():
    try:
        limit = int(request.args.get('limit', 50))
        past_predictions = get_history(limit=limit)
        return jsonify({"history": past_predictions}), 200
    except Exception as e:
        logger.error(f"Error fetching history: {traceback.format_exc()}")
        return jsonify({"error": "Failed to fetch history."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
