#import model # Import the python file containing the ML model
from flask import Flask, request, render_template # Import flask libraries
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
#import pickle
import numpy as np
import joblib
from joblib import load

# Initialize the flask class and specify the templates directory
app = Flask(__name__,template_folder="templates")
engine = create_engine("postgresql://postgres:admin@localhost:5432/postgres")
db = scoped_session(sessionmaker(bind=engine))
app.secret_key = '12345678' # this key is used to communicate with database.
#Configure session to use filesystem
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
# Default route set as 'home'
@app.route('/')
def home():
    return render_template('index.html') # Render home.html

#prediction call
@app.route('/predict',methods=['POST','GET'])
def prediction():
    # standardisation call
    x_sc1=load('x_std_scaler.bin')
    y_sc1=load('y_std_scaler.bin')
    # get the input data columns which are needed for prediction
    # and performing preprocessing actions
    age = request.args['age']
    gender = request.args['gender']
    if gender=="male":
        gender1=1
    else:
        gender1=0
    smoker = request.args['smoker']
    if smoker=="yes":
        smoker1=1
    else:
        smoker1=0
    bmi = request.args['bmi']    
    noc = request.args['noc']        
    Region = request.args['Region']
    if Region=="northeast":
        Region1 =0
        Region2 =0
        Region3 =0
    if Region=="northwest":
        Region1 =1
        Region2 =0
        Region3 =0
    if Region=="southeast":
        Region1 =0
        Region2 =1
        Region3 =0
    if Region=="southwest":
        Region1 =0
        Region2 =0
        Region3 =1
    ds=np.array([age,gender1,bmi,noc,smoker1,Region1,Region2,Region3])
    #standardisation of input features
    ds1=x_sc1.transform(ds.reshape(-1,8))
    # SVM model loading
    model=joblib.load('model.pkl')
    # prediction call and inverse transform to get exact value
    predicted_value=y_sc1.inverse_transform(model.predict(ds1))
    print(predicted_value)
    #result =round(predicted_value[0],3)
    amount=round(predicted_value[0],2)
    # insert the data into postgres
    db.execute("INSERT INTO insurance (age, gender,bmi, noc,region,smoker,amount) VALUES (:age,:gender,:bmi,:noc,:region,:smoker,:amount)",
            {"age": age, "gender": gender,"bmi":bmi,"noc":noc,"smoker":smoker,"region":Region,"amount":amount}) 
    db.commit() 
    return render_template('output.html', result=round(predicted_value[0],2),age=age,gender=gender,bmi=bmi,noc=noc,smoker=smoker,region= Region)

# display call
@app.route('/disp')
def home_display():
    return render_template('displayall.html') # Render home.html

# no condition, simple call for getting all data
@app.route("/displayall",methods=['POST','GET'])
def list():
   details=db.execute("select * from insurance")
   return render_template("display.html", details=details)

#based on condition ,data retrieval
@app.route("/display",methods=['POST','GET'])
def listall():
   age = request.args['age']
   gender = request.args['gender']
   smoker = request.args['smoker']
   bmi = request.args['bmi']
   noc = request.args['noc']      
   Region = request.args['Region']
   details=db.execute("select * from insurance where age=:age and gender=:gender and smoker=:smoker and bmi=:bmi and noc=:noc and Region =:Region;", {"age":age,"gender":gender,"smoker":smoker,"bmi":bmi,"noc":noc,"Region":Region})
   return render_template("display.html", details=details)

if(__name__=='__main__'):
  #classify_type()
  app.run(debug=True)   
	