from flask import Flask , render_template, request
from joblib import load

app = Flask(__name__)

# Load our pre-trained model
clf = load('./model/iris_classifier.joblib')

@app.route("/sub", methods = ["POST"])
def submit():
    # HTML -> .py
    if request.method == "POST":
        s_l = request.form["sepal_length"]
        s_w = request.form["sepal_width"]
        p_l = request.form["petal_length"]
        p_w = request.form["petal_width"]
    prediction = clf.predict([[s_l, s_w, p_l, p_w]])
    # Map the predicted value to an actual class
    if prediction[0] == 0:
        predicted_class = "Iris-Setosa"
    elif prediction[0] == 1:
        predicted_class = "Iris-Versicolour"
    else:
        predicted_class = "Iris-Virginica"
    #.py -> HTML
    return render_template("sub.html", prediction = predicted_class)


@app.route("/")
def hello():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
