from flask import Flask, send_file

app = Flask(__name__)

@app.route('/')
def home():
    return "Entrenando red neuronal! Descarga los pesos más recientes desde /w"

@app.route('/w')
def send_weights():
    w_fn = "save/2048ddqn2.h5"
    return send_file(w_fn, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
