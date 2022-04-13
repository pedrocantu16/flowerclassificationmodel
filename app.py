from flask import Flask , render_template, request
from joblib import load
from PIL import Image
import base64
import io

app = Flask(__name__)

# Load our pre-trained model
clf = load('./model/iris_classifier.joblib')

@app.route("/sub", methods = ["POST"])
def submit():
    im = None
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
        im = Image.open("./image/iris_setosa.jpeg")
    elif prediction[0] == 1:
        predicted_class = "Iris-Versicolour"
        im = Image.open("./image/iris_versicolor.jpeg")
    else:
        predicted_class = "Iris-Virginica"
        im = Image.open("./image/iris_virginica.jpeg")
    
    # Get the in-memory info using below code line.
    data = io.BytesIO()
    #First save image as in-memory.
    im.save(data, "JPEG")
    #Then encode the saved image file.
    encoded_img_data = base64.b64encode(data.getvalue())

    #.py -> HTML
    return render_template("sub.html", prediction = predicted_class, img_data=encoded_img_data.decode('utf-8'))


@app.route("/")
def hello():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
