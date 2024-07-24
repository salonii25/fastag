import pandas as pd
import numpy as np
import gradio as gr
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def train_model():
    df = pd.read_csv('FastagFraudDetection.csv')
    df['FastagID'] = df['FastagID'].fillna(df['FastagID'].mode()[0])
    df.drop_duplicates(inplace=True)

    label_encoders = {}
    for column in ['Vehicle_Type', 'Lane_Type', 'Vehicle_Dimensions', 'Geographical_Location',
                   'Vehicle_Plate_Number', 'Fraud_indicator', 'FastagID', 'TollBoothID']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    df.drop(columns=['Timestamp'], inplace=True)

    x = df.drop(columns=['Fraud_indicator','Geographical_Location','Vehicle_Speed','Transaction_Amount','Amount_paid','Transaction_ID'])
    y = df['Fraud_indicator']

    

    model = DecisionTreeClassifier()
    model.fit(x, y)

    return model, label_encoders

model, label_encoders = train_model()

def main(Vehicle_Type, FastagID, TollBoothID, Lane_Type, Vehicle_Dimensions, Vehicle_Plate_Number):
    features = [
        label_encoders['Vehicle_Type'].transform([Vehicle_Type])[0],
        label_encoders['FastagID'].transform([FastagID])[0],
        label_encoders['TollBoothID'].transform([TollBoothID])[0],
        label_encoders['Lane_Type'].transform([Lane_Type])[0],
        label_encoders['Vehicle_Dimensions'].transform([Vehicle_Dimensions])[0],
        label_encoders['Vehicle_Plate_Number'].transform([Vehicle_Plate_Number])[0],
    ]
    
    features = np.array(features).reshape(1, -1)
    pred = model.predict(features)[0]

    result = "Fraud" if pred == 0 else "Not Fraud"

    return result

fraud_detector = gr.Interface(
    fn=main,
    inputs=[
        gr.Textbox(label="Vehicle Type"),
        gr.Textbox(label="FastagID"),
        gr.Textbox(label="TollBoothID"),
        gr.Textbox(label="Lane Type"),
        gr.Textbox(label="Vehicle Dimensions"),
        gr.Textbox(label="Vehicle Plate Number")
    ],
    outputs=gr.Textbox(label="Fraud Indicator")
)

fraud_detector.launch()
