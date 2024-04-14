from flask import Flask
from flask_cors import CORS
from flask import request
from flask import jsonify, render_template #,redirect, url_for,
#from requests import get

app = Flask(__name__)
CORS(app)

@app.get("/")
def index_get():
    return render_template("index.html")

#@app.route('/', methods=["POST", "GET"])
@app.post("/predict")
def predict(): 
    from chat import get_response
    from chat import predict_class, intents
    text = request.get_json().get("message")
    #s = request
    ints=predict_class(text)
    response = get_response(ints, intents)
    message = {"answer": response}
    return jsonify(message)

if __name__ == '__main__':
    app.run(debug=True)

#rename fromtend folder to templates
'''
@app.route('/welcome')
def welcome:
    return render_template('index.html')

@app.route('/health', methods=["POST", "GET"])
def health():
    return 
@app.route('/dashboard')
def dashboard():
    return "This is the dashboard"

def red():
    return redirect(url_for(war, nowar))

'''
###url for separates link by /


