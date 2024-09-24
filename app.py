from flask import Flask,request, url_for, redirect, render_template
import joblib
import pandas as pd
import sqlite3
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)

loaded_model1 = joblib.load("model.sav")

data = pd.read_csv('data/data.csv')

from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
data['Name']= label_encoder.fit_transform(data['Name'])
data['Gender']= label_encoder.fit_transform(data['Gender'])
data['Disease']= label_encoder.fit_transform(data['Disease'])

data['Name'].unique()
data['Gender'].unique()
data['Disease'].unique()

X = data.iloc[:, 2:7]
y = data.iloc[:,7]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

model = RandomForestClassifier(n_estimators = 100, oob_score = 'TRUE', n_jobs = -1,random_state =50,max_features = "auto", min_samples_leaf = 50)
model.fit(X,y)
predictions = model.predict(X_test)

@app.route('/')
def hello_world():
    return render_template("home.html")

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route('/note')
def note():
	return render_template('notebook.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    text1 = request.form['1']
    text2 = request.form['2']
    text3 = request.form['3']
    text4 = request.form['4']
    text5 = request.form['5']
    text6 = request.form['6']
    
 
    row_df = pd.DataFrame([pd.Series([text2,text3,text4,text5,text6])])
    print(row_df)
    #prediction=model.predict(row_df)
    prediction= loaded_model1.predict(row_df)
    output=round(prediction[0],2)
    print(output)   
    if output == 0:
        result = 'CVD'
        if str(text2) > str(18):
            
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `Disease`,`Age2` from drug where `Disease` = ?",(result,))
            data = cur.fetchall()
            print(data)
            return render_template('result.html',ans='CardivasularDisease-CVD', output_data=data)
        else:
            
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `Disease`,`Age1` from drug where `Disease` = ?",(result,))
            data = cur.fetchall()
            print(data)
            return render_template('result.html',ans='CardivasularDisease-CVD', output_data=data)

    elif output == 1:
        result = 'Chronic kidney disease'

        if str(text2) > str(18):
            
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `Disease`,`Age2` from drug where `Disease` = ?",(result,))
            data = cur.fetchall()
            print(data)
            return render_template('result.html',ans='Chronic kidney disease', output_data=data)
        else:
            
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `Disease`,`Age1` from drug where `Disease` = ?",(result,))
            data = cur.fetchall()
            print(data)
            return render_template('result.html',ans='Chronic kidney disease', output_data=data)
    elif output == 2:
        result = 'Diabetes'
        if str(text2) > str(18):
            
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `Disease`,`Age2` from drug where `Disease` = ?",(result,))
            data = cur.fetchall()
            print(data)
            return render_template('result.html',ans='Diabetes', output_data=data)
        else:
            
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `Disease`,`Age1` from drug where `Disease` = ?",(result,))
            data = cur.fetchall()
            print(data)
            return render_template('result.html',ans='Diabetes', output_data=data)
    elif output == 3:
        result = 'Eating disorder'
        if str(text2) > str(18):
            
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `Disease`,`Age2` from drug where `Disease` = ?",(result,))
            data = cur.fetchall()
            print(data)
            return render_template('result.html',ans='Eating disorder', output_data=data)
        else:
            
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `Disease`,`Age1` from drug where `Disease` = ?",(result,))
            data = cur.fetchall()
            print(data)
            return render_template('result.html',ans='Eating disorder', output_data=data)
    elif output == 4:
        result = 'Heart Condition'
        if str(text2) > str(18):
            
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `Disease`,`Age2` from drug where `Disease` = ?",(result,))
            data = cur.fetchall()
            print(data)
            return render_template('result.html',ans='Heart Condition', output_data=data)
        else:
            
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `Disease`,`Age1` from drug where `Disease` = ?",(result,))
            data = cur.fetchall()
            print(data)
            return render_template('result.html',ans='Heart Condition', output_data=data)
    elif output == 5:
        result = 'Hypertension'
        if str(text2) > str(18):
            
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `Disease`,`Age2` from drug where `Disease` = ?",(result,))
            data = cur.fetchall()
            print(data)
            return render_template('result.html',ans='Hypertension', output_data=data)
        else:
            
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `Disease`,`Age1` from drug where `Disease` = ?",(result,))
            data = cur.fetchall()
            print(data)
            return render_template('result.html',ans='Hypertension', output_data=data)
    elif output == 6:
        result = 'Obesity'
        if str(text2) > str(18):
            
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `Disease`,`Age2` from drug where `Disease` = ?",(result,))
            data = cur.fetchall()
            print(data)
            return render_template('result.html',ans='Obesity', output_data=data)
        else:
            
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("select `Disease`,`Age1` from drug where `Disease` = ?",(result,))
            data = cur.fetchall()
            print(data)
            return render_template('result.html',ans='Obesity', output_data=data)
    else:
        result = 'NULL'
        data = 'Invaild'
        return render_template('result.html',ans=result,pred=f'You are safe.\n Probability of having Disease is {output}',output_data=data)

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/index')
def index():
	return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
