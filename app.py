from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load GPT-2 Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Predefined knowledge base
knowledge_base = {
    "what is python": "Python is a popular programming language.",
    "who is the president of the USA": "Joe Biden is the president of the United States.",
    "what is AI": "AI (Artificial Intelligence) is the simulation of human intelligence in machines."
}


@app.route("/")
def home():
    return render_template("index.html")  # Serve the frontend


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").lower()

    if not user_input:
        return jsonify({"response": "Please provide a message!"})

    # Search in the knowledge base for a relevant answer
    response = knowledge_base.get(user_input, None)

    if response:
        return jsonify({"response": response})

    # If not found in knowledge base, use GPT-2 for response generation
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=100, temperature=0.7, top_p=0.9,
                                  pad_token_id=tokenizer.eos_token_id)
    chatbot_reply = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return jsonify({"response": chatbot_reply})


if __name__ == "__main__":
    app.run(debug=True)
