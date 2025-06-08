from flask import Flask,render_template,request,redirect
import pickle
import numpy as np

model=pickle.load(open("model.pkl","rb"))

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict_admission():
    GRE=float(request.form.get("GRE"))
    TOEFL=float(request.form.get("TOEFL"))
    Uni_rating=float(request.form.get("Uni_rating"))
    SOP=float(request.form.get("SOP"))
    LOR=float(request.form.get("LOR"))
    CGPA=float(request.form.get("CGPA"))
    Research=float(request.form.get("Research"))

    
    
    result=model.predict(np.array([[GRE,TOEFL,Uni_rating,SOP,LOR,CGPA,Research]]))
    return render_template('index.html' , chance_of_admit = 'student has a {} probability to get admission'.format(result))

app.run(host="127.0.0.1",port=5001,debug=True)