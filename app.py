from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

# Tải các mô hình
models = {}
model_names = ['linear', 'ridge', 'mlp', 'stacking']
for model_name in model_names:
    with open(f'models/{model_name}_model.pkl', 'rb') as f:
        models[model_name] = pickle.load(f)

# Khởi tạo scaler và lấy giá trị mean và std từ mô hình đã huấn luyện
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            data = request.get_json()
            tv = float(data['tv'])
            radio = float(data['radio'])
            newspaper = float(data['newspaper'])
            selected_model = data['model']

            input_data = np.array([[tv, radio, newspaper]])
            input_data_scaled = scaler.transform(input_data)

            prediction = models[selected_model].predict(input_data_scaled)[0]
            prediction = round(prediction * 1000, 2)

            return {'prediction': prediction}

        except Exception as e:
            return {'error': str(e)}

    return render_template('index.html', model_names=model_names)

if __name__ == '__main__':
    app.run(debug=True)
