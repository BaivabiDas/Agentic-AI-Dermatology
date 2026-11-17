
import ollama
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "Recommendation Agent is running"
    })


@app.route("/recommend", methods=["POST"])
def recommend():

    data = request.get_json(silent=True) or {}
    disease = data.get("disease")

    if not disease:
        return jsonify({
            "status": "error",
            "message": "Disease name is required"
        }), 400

    prompt = f"""
You are a dermatology recommendation assistant.

A patient has a predicted skin condition: {disease}

Provide a SHORT response in this format:

Disease:
Symptoms:
- symptom 1
- symptom 2

Homeopathic Remedies:
- remedy 1
- remedy 2

Advice:
- advice 1
- advice 2

Maximum 80 words.
Do not give long explanations.
Clearly state that this is not a medical diagnosis
and that professional medical advice should be sought.
"""

    try:
        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return jsonify({
            "status": "success",
            "disease": disease,
            "recommendation": response["message"]["content"]
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5002
    )
