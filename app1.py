# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 14:37:33 2022

@author: leonardo
"""
#importación de librerias necesarias 
#para desplegar la aplicación en una interfaz web
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from sklearn import svm
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#Se definen dos funciones para cada modelo.
#Cada funcion recibe como parametros lo valores ingresados por los usuarios
# y el modelo entrenado
def model_Logistica(x_in, RegLogist):
    x = np.asarray(x_in).reshape(1,-1)
    preds=RegLogist.predict(x)
    return preds

def model_bayes(x_in,NaBay):
    x = np.asarray(x_in).reshape(1,-1)
    preds=NaBay.predict(x)
    return preds



#Ruras donde estan guardads los modelos entrenados
Ruta1 = 'models/Regresión_Logistica.pkl'
Ruta2 = 'models/Naive_Bayes.pkl'
RegLogist=''
NaBay=''
with open(Ruta1, 'rb') as Lo:
    RegLogist = pickle.load(Lo)
with open(Ruta2, 'rb') as Ba:
    NaBay = pickle.load(Ba)
    
    
def main():
    st.title('Modelos para predecir la probabilidad de aprobar el último parcial'+
              ' de los estudiantes')
    st.sidebar.header('Prediccion y Modelos')
    #parametros de ingreso de datos por el usirario
    def parametros_usario():
        estudio= st.sidebar.slider('Tiempo de Estudio',1,4,2)
        romantico= st.sidebar.slider('Situación Romantica',0,1,0)
        salud= st.sidebar.slider('Estado de salud',1,5,2)
        ausencias= st.sidebar.slider('Numero de ausencias',0,93,2)
        G1= st.sidebar.slider('Nota del parcial 1',0,20,1)
        G2= st.sidebar.slider('Nota del parcial 2',0,20,3)
        #Se almacenan los parametros en un diccionario
        datos={'T. estudio':estudio,
               'Situación Romantico':romantico,
               'Salud':salud,
               '# ausencias':ausencias,
               'Parcial 1':G1,
               'Parcial 2':G2
               }
        
        parametros=pd.DataFrame(datos, index=[0])
        #se retornar los prametros
        return parametros
    #se llaman y visualizan los parametros
    df=parametros_usario()
    #Opciones para escoger el modelo
    Opciones_modelos=['Regresion Logística', 'Naive Bayes']
    modelos=st.sidebar.selectbox('Elija el modelo con el que quiere predecir', Opciones_modelos)
    
    
    #datos escogidos por el Usario
    st.subheader('Datos escogidos por el usuario')
    st.subheader(modelos)
    st.write(df)
    #boton la predicción
    if st.button('Predecir'):
        #Si se escoge la Regesion Logistica se presentara los datos de la prediccion
        if modelos=='Regresion Logística':
            predictS = model_Logistica(df, RegLogist)
            #st.success('EL ESTUDIANTE: {}'.format(predictS[0]).upper())
            prediccion=predictS[0]
            x_i=np.asarray(df).reshape(1,-1)
            #si el valor es 1 aprueba
            if prediccion ==1:
                st.success('El Estudiante: APRUEBA')
                probabilidad=RegLogist.predict_proba(x_i)
                #imprime probabilidad deaprobar
                st.success('La Probabilidad de aprobar es : :{}'.format(probabilidad[:,1]*100))
            else:
                st.success('El Estudiante: REPRUEBA')
                probabilidad=RegLogist.predict_proba(x_i)
                st.success('La Probabilidad de aprobar es :{}'.format(probabilidad[:,1]*100))
        else:
            #Si se escoge Naive Bayes se presentara los datos de la prediccion
            predictS1 = model_bayes(df, NaBay)
            #st.success('EL ESTUDIANTE: {}'.format(predictS[0]).upper())
            prediccion=predictS1[0]
            x_i=np.asarray(df).reshape(1,-1)
            if prediccion ==1:
                st.success('El Estudiante: APRUEBA')
                probabilidad=NaBay.predict_proba(x_i)
                st.success('La Probabilidad de aprobar es :{}'.format(probabilidad[:,1]*100))
            else:
                st.success('El Estudiante: REPRUEBA')
                probabilidad=NaBay.predict_proba(x_i)
                st.success('La Probabilidad de aprobar es :{}'.format(probabilidad[:,1]*100))
if __name__=='__main__':
    main()