import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from sklearn import svm
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Rutas de los modelos preentrenado
Ruta1 = 'models/Regresión_Logistica.pkl'
Ruta2 = 'models/Naive_Bayes.pkl'
# Se recibe la imagen y el modelo, devuelve la predicción
def model_prediction1(x_in, model1):
    x = np.asarray(x_in).reshape(1,-1)
    preds=model1.predict(x)
    return preds


def model_bayes(x_in,model2):
    x = np.asarray(x_in).reshape(1,-1)
    preds=model2.predict(x)
    return preds
def main():
    
    model1=''
    model2=''

    # Se carga el modelo
    if model1=='':
        with open(Ruta1, 'rb') as file:
            model1 = pickle.load(file)
    if model2=='':
         with open(Ruta2, 'rb') as file:
             model2 = pickle.load(file)
    
    # Título
    html_temp = """
    <h1 style="color:#181082;text-align:center;">SISTEMA PARA PREDECIR EL ESTADO DEL ESTUDIANTE EN EL PARCIAL FINAL </h1>
    <body>
	INSTRUCCIONES: Todos los valores a ingresar deben ser numericos
        <ul>
            <li>Tiempo de estudio (en horas 1 a 4) </li>
            <li>Estado de salud (muy malo-1 |||muy bueno-5)</li>
            <li>Ausencia eScolar (0-90)</li>
            <li>Nota del parcial 1 y 2( de 0 a 20)</li>
            <li>Situacion setimental ( si-1 ||| no-0 )</li>
        </ul>
	</body>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    # Lecctura de datos
    #Datos = st.text_input("Ingrese los valores : N P K Temp Hum pH lluvia:")
    N = st.text_input("Tiempo de estudio:")
    R = st.text_input("En relacion Sentimental")
    E = st.text_input("Estado de salud:")
    P = st.text_input("Ausencia:")
    K = st.text_input("Nota Parcial 1:")
    No = st.text_input("Nota Parcial 2:")
    

    # El botón predicción se usa para iniciar el procesamiento
    if st.button("PredicciónLogistica :"): 
        #x_in = list(np.float_((Datos.title().split('\t'))))
        x_in =[np.float_(N.title()),
               np.float_(R.title()),
               np.float_(E.title()),
               np.float_(P.title()),
               np.float_(K.title()),
               np.float_(No.title())
                    ]
        predictS = model_prediction1(x_in, model1)
        #st.success('EL ESTUDIANTE: {}'.format(predictS[0]).upper())
        prediccion=predictS[0]
        x_i=np.asarray(x_in).reshape(1,-1)
        if prediccion ==1:
            st.success('El Estudiante: APRUEBA')
            probabilidad=model1.predict_proba(x_i)
            st.success('La Probabilidad de aprobar es : :{}'.format(probabilidad[:,1]*100))
        else:
            st.success('El Estudiante: REPRUEBA')
            probabilidad=model1.predict_proba(x_i)
            st.success('La Probabilidad de aprobar es :{}'.format(probabilidad[:,1]*100))
    
if __name__ == '__main__':
    main()
    
