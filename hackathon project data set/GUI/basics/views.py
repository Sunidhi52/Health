from django.shortcuts import render

# Create your views here.

def index(request):
    return render(request, 'index.html')

def diabetes(request):
    import pandas as pd
    path="C:\\Users\\sunid\\Desktop\\hackathon project data set\\diabetes_prediction_dataset.csv"
    data=pd.read_csv(path)
    data['smoking_history']=data['smoking_history'].map({'never':0,'No Info':1,'current':2,'former':3,'ever':4,'not current':5})
    data['gender']=data['gender'].map({'Female':0,'Male':1,'Other':2})
    inputs=data.drop('diabetes',axis=1)
    outputs=data.drop(['hypertension','gender','age','smoking_history','bmi','HbA1c_level','blood_glucose_level','heart_disease'],axis=1)

    import sklearn
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(inputs,outputs,test_size=0.2)
    
    from sklearn.ensemble import RandomForestClassifier  #randomforest algorithm used
    model=RandomForestClassifier(n_estimators=200)
    model.fit(x_train,y_train)

    if(request.method=="POST"):
        data = request.POST
        hypertension = data.get('hypertension')
        gender = data.get('gender')
        age = data.get('age')
        smoking_history = data.get('smoking_history')
        bmi = data.get('bmi')
        HbA1c_level = data.get('HbA1c_level')
        blood_glucose_level = data.get('blood_glucose_level')
        heart_disease = data.get('heart_disease')
        result = model.predict([[float(hypertension),float(gender),float(age),float(smoking_history),float(bmi),float(HbA1c_level),float(blood_glucose_level),float(heart_disease)]])
        if result==0:
            return render(request,'diabetes.html',context={'outcome':'you do not have diabetes'})
        elif result==1:
            return render(request,'diabetes.html',context={'outcome':'you have diabetes'})
        else:
            return render(request,'diabetes.html',context={'outcome':'check your inputs and give valid entries'})
        return render(request,'diabetes.html',context={'result':result[0]})
    return render(request,'diabetes.html')

def sleep(request):

    import pandas as pd
    from django.shortcuts import render
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Load and prepare data
    path = "C:\\Users\\sunid\\Desktop\\hackathon project data set\\Sleep_health_and_lifestyle_dataset.csv"
    data = pd.read_csv(path)
    data = data.drop('Person ID', axis=1)
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    data['Occupation'] = data['Occupation'].map({'Software Engineer': 0, 'Doctor': 1, 'Sales Representative': 2, 'Teacher': 3, 'Nurse': 4})
    data['Sleep Disorder'] = data['Sleep Disorder'].map({'None': 0, 'Sleep Apnea': 1, 'Insomnia': 2, 'former': 3})
    data['BMI Category'] = data['BMI Category'].map({'Overweight': 0, 'Normal': 1, 'Obese': 2, 'former': 3, 'Normal Weight': 4})
    data['Occupation'] = data['Occupation'].fillna(data['Occupation'].median())

    inputs = data.drop('Sleep Disorder', axis=1)
    outputs = data['Sleep Disorder']

    # Handle NaN values in the target variable
    outputs = outputs.fillna(outputs.median())

    # Train the model
    x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    if request.method == "POST":
        data = request.POST
        Gender = data.get('Gender')
        Age = data.get('Age')
        Occupation = data.get('Occupation')
        Sleep_Duration = data.get('Sleep_Duration')
        Quality_of_Sleep = data.get('Quality_of_Sleep')
        Physical_Activity_Level = data.get('Physical_Activity_Level')
        Stress_Level = data.get('Stress_Level')
        BMI_Category = data.get('BMI_Category')
        Heart_Rate = data.get('Heart_Rate')
        Daily_Steps = data.get('Daily_Steps')
        Systolic_Pressure = data.get('Systolic_Pressure')
        Diastolic_Pressure = data.get('Diastolic_Pressure')

        features = [float(Gender), float(Age), float(Occupation), float(Sleep_Duration), float(Quality_of_Sleep),
                    float(Physical_Activity_Level), float(Stress_Level), float(BMI_Category), float(Heart_Rate),
                    float(Daily_Steps), float(Systolic_Pressure), float(Diastolic_Pressure)]

        result = model.predict([features])[0]

        if result == 0:
            outcome = 'you do no have sleep disorder'
        elif result == 1:
            outcome = 'you have sleep apnea'
        elif result == 2:
            outcome = 'you have insomnia'
        elif result == 3:
            outcome = 'you formerly had sleep'
        else:
            outcome = 'check your inputs and give valid entries'

        return render(request, 'sleep.html', context={'outcome': outcome})
    
    return render(request, 'sleep.html')

from django.shortcuts import render
from django import forms

# Form to capture height and weight
class BMIForm(forms.Form):
    height = forms.FloatField(label='Height (in meters)', min_value=0.1)
    weight = forms.FloatField(label='Weight (in kilograms)', min_value=0.1)

def calculate_bmi(request):
    bmi = None
    bmi_category = None

    if request.method == 'POST':
        form = BMIForm(request.POST)
        if form.is_valid():
            height = form.cleaned_data['height']
            weight = form.cleaned_data['weight']

            # Calculate BMI
            bmi = weight / (height ** 2)

            # Determine BMI Category
            if bmi < 18.5:
                bmi_category = 'Underweight'
            elif 18.5 <= bmi < 24.9:
                bmi_category = 'Normal weight'
            elif 25 <= bmi < 29.9:
                bmi_category = 'Overweight'
            else:
                bmi_category = 'Obese'

            bmi = round(bmi, 2)
    else:
        form = BMIForm()

    # Render the form and result in the same template
    return render(request, 'bmi_calculator.html', {
        'form': form,
        'bmi': bmi,
        'bmi_category': bmi_category,
    })



