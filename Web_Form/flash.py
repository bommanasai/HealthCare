from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

data = pd.read_csv('healthcare.csv')

data.drop(columns=['Date of Admission','Doctor', 'Hospital', 'Insurance Provider','Billing Amount','Room Number', 'Admission Type', 'Discharge Date',
       'Medication','Test Results'],inplace=True)

le = LabelEncoder()
data['Blood Group Type'] = le.fit_transform(data['Blood Group Type'])
data['Gender'] = le.fit_transform(data['Gender'])

X = data.drop(columns=['Medical Condition', 'Name'])
y = data['Medical Condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    age = int(request.form['age'])
    gender = request.form['gender']
    blood_group = request.form['blood_group']

    gender_encoded = le.fit_transform([gender])[0]
    blood_group_encoded = le.fit_transform([blood_group])[0]
    
    prediction = clf.predict([[age,gender_encoded, blood_group_encoded]])
    save_to_csv(name, gender, blood_group)
    return render_template('result.html', name=name, prediction=prediction)

def save_to_csv(name, gender, blood_group):
    data = {'Name': [name], 'Gender': [gender], 'Blood Group': [blood_group]}
    df = pd.DataFrame(data,columns=data)
    df.to_csv('users.csv',mode='a', index=False, header=False)

if __name__ == '__main__':
    app.run(debug=True)
