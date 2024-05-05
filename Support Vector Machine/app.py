from flask import Flask, request, render_template
import joblib
import pdfplumber
from bs4 import BeautifulSoup
import email
from email import policy

app = Flask(__name__)

# Load your trained model and vectorizer
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


def extract_text_from_eml(file):
    msg = email.message_from_binary_file(file, policy=policy.default)
    texts = []
    for part in msg.walk():
        ctype = part.get_content_type()
        cdispo = str(part.get('Content-Disposition'))

        if ctype == 'text/plain' and 'attachment' not in cdispo:
            texts.append(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
        elif ctype == 'text/html':
            html = part.get_payload(decode=True).decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html, "html.parser")
            texts.append(soup.get_text())
    return ' '.join(texts)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        email_text = None
        if 'email_file' in request.files and request.files['email_file'].filename != '':
            file = request.files['email_file']
            if allowed_file(file.filename):
                if file.filename.lower().endswith('.pdf'):
                    with pdfplumber.open(file) as pdf:
                        pages = [page.extract_text() for page in pdf.pages]
                    email_text = "\n".join(filter(None, pages))
                elif file.filename.lower().endswith('.eml'):
                    email_text = extract_text_from_eml(file.stream)
                else:
                    email_text = file.read().decode('utf-8')
        elif 'email_text' in request.form and request.form['email_text'].strip():
            email_text = request.form['email_text']

        if email_text:
            vectorized_text = vectorizer.transform([email_text])
            prediction = model.predict(vectorized_text)
            message = "Safe Email" if prediction[0] == 1 else "Phishing Email"
            return render_template('index.html', prediction=message)
        else:
            return render_template('index.html', prediction="Please enter text or upload a valid file.")
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['txt', 'eml', 'pdf']


if __name__ == '__main__':
    app.run(debug=True)
