#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('knn_breast_cancer_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Clump_thickness   = float(request.form['Clump_thickness'])
    Uniformity_of_cell_size   = float(request.form['Uniformity_of_cell_size'])
    Uniformity_of_cell_shape   = float(request.form['Uniformity_of_cell_shape'])
    Marginal_adhesion     = float(request.form['Marginal_adhesion'])
    Single_epithelial_cell_size     = float(request.form['Single_epithelial_cell_size'])
    Bare_nuclei     = float(request.form['Bare_nuclei'])
    Bland_chromatin     = float(request.form['Bland_chromatin'])
    Normal_nucleoli       = float(request.form['Normal_nucleoli'])
    Mitoses       = float(request.form['Mitoses'])

    features = np.array([[Clump_thickness, Uniformity_of_cell_size, Uniformity_of_cell_shape,Uniformity_of_cell_shape,Marginal_adhesion,
                         Single_epithelial_cell_size,Bare_nuclei,Bland_chromatin,Normal_nucleoli,Mitoses]])
    prediction = model.predict(features)
    return f'Predicted benign (label = 0) or malignant (label = 1) code: {prediction[0]}'

if __name__ == '__main__':
    app.run(debug=True)

