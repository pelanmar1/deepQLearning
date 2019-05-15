from flask import Flask

app = Flask(__name__,static_url_path='/save')


@app.route('/')
def hello():
    return "Hello World!"


@app.route('/w')
def hello_name():
    w_fn = "temp_w.h5"
    return app.send_static_file(w_fn)

if __name__ == '__main__':
    app.run()