from flask import Flask, render_template
from MLR import * 

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/salvador")
def run():
    return 42


if __name__ == "__main__":
    app.run(debug=True)
