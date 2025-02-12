import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load GPT-2 Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

@app.route("/")
def home():
    return render_template("index.html")  # Serve the frontend

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"response": "Please provide a message!"})

    # Encode input and generate response
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=100, temperature=0.7, top_p=0.9,
                                  pad_token_id=tokenizer.eos_token_id)

    chatbot_reply = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return jsonify({"response": chatbot_reply})

if __name__ == "__main__":
    # Use the port from the environment variable or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
