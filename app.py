import http.server
import socketserver
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import py_vncorenlp
import re
import numpy as np

# Lấy đường dẫn tới thư mục hiện tại
BASE_DIR = Path(__file__).resolve().parent

# Khởi tạo VnCoreNLP
vncorenlp_dir = BASE_DIR / "vncorenlp"
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(vncorenlp_dir))

# Định nghĩa các ký tự đặc biệt cần thay thế
special_chars = {
    ":)": "colonsmile",
    ":(": "colonsad",
    "@@": "colonsurprise",
    "<3": "colonlove",
    ":d": "colonsmilesmile",
    ":3": "coloncontemn",
    ":v": "colonbigsmile",
    ":_": "coloncc",
    ":p": "colonsmallsmile",
    ">>": "coloncolon",
    ':">': "colonlovelove",
    "^^": "colonhihi",
    ":": "doubledot",
    ":'(": "colonsadcolon",
    ":@": "colondoublesurprise",
    "v.v": "vdotv",
    "...": "dotdotdot",
    "/": "fraction",
    "c#": "csharp"
}

# Hàm để thay thế các ký tự đặc biệt trong văn bản
def replace_special_chars(text):
    for char, replacement in special_chars.items():
        text = re.sub(re.escape(char), replacement, text, flags=re.IGNORECASE)
    return text

# Hàm để thực hiện word segmentation trên văn bản
def word_segmentation(text):
    segmented_text = rdrsegmenter.word_segment(text)
    return " ".join(segmented_text)

# Tải mô hình và tokenizer
class CustomBERT(torch.nn.Module):
    def __init__(self):
        super(CustomBERT, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2", num_labels=3, output_hidden_states=True)
        self.classifier = torch.nn.Linear(768 * 4, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[-4:]
        cls_outputs = torch.cat([hidden_states[i][:, 0, :] for i in range(4)], dim=1)
        logits = self.classifier(cls_outputs)
        return logits

model = CustomBERT()
model.load_state_dict(torch.load(BASE_DIR / 'pytorch_model.bin', map_location=torch.device('cpu')))
model.eval()

tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR / 'tokenizer'))

# Hàm dự đoán
def predict_multiple(texts):
    model.eval()
    predictions = []
    probabilities = []

    for text in texts:
        # Tiền xử lý văn bản
        processed_text = replace_special_chars(text)
        segmented_text = word_segmentation(processed_text)

        # Mã hóa văn bản
        inputs = tokenizer(segmented_text, padding=True, truncation=True, return_tensors="pt")

        # Dự đoán
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            prob = torch.softmax(outputs, dim=1).cpu().numpy()

        # Lưu lớp dự đoán và xác suất
        predicted_class = np.argmax(prob, axis=1)[0]
        predictions.append(predicted_class)
        probabilities.append(prob[0])

    return predictions, probabilities

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            text = data['text']
            
            predictions, probabilities = predict_multiple([text])
            
            sentiment_map = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
            result = {
                "sentiment": sentiment_map[predictions[0]],
                "probabilities": probabilities[0].tolist()
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

PORT = 8000

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()