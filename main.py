from flask import Flask, request, jsonify, make_response
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Загрузка модели и токенизатора
model_path = "model_anekdots"  # Указывайте актуальный путь к модели
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    input_text = data.get('text', '')  # Получаем текст, используя .get с пустой строкой в качестве значения по умолчанию

    # Проверяем, не пустой ли ввод
    if not input_text.strip():  # .strip() удаляет пробельные символы в начале и конце строки
        return jsonify({'generated_text': 'Пожалуйста, введите текст для генерации.'}), 400

    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Проверяем, не пустой ли тензор
    if input_ids.nelement() == 0:  # Если нет элементов в input_ids
        return jsonify({'generated_text': 'Ошибка при обработке введенного текста.'}), 400

    model.eval()
    with torch.no_grad():
        out = model.generate(input_ids, do_sample=True, num_beams=3, temperature=1.7,
                             top_p=0.95, max_length=70, repetition_penalty=1.2)

    generated_text = tokenizer.decode(out[0], skip_special_tokens=True)
    response = make_response(jsonify({'generated_text': generated_text}))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


if __name__ == '__main__':
    app.run(debug=True, port=8080)
