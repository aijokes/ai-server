from flask import Flask, request, jsonify
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

# Загрузка модели и токенизатора
model_path = "model_anekdots"  # Обновите путь к вашей модели и токенизатору
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    input_text = data['text']
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        out = model.generate(input_ids, do_sample=True, num_beams=3, temperature=1.7,
                             top_p=0.95, max_length=70, repetition_penalty=1.2)
    generated_text = tokenizer.decode(out[0], skip_special_tokens=True)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)
