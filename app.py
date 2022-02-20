# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 14:37:33 2022

@author: leonardo
"""
# importación de librerias necesarias
# para desplegar la aplicación en una interfaz web
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sb 
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from PIL import Image

# Se definen dos funciones para cada modelo.
# Cada funcion recibe como parametros lo valores ingresados por los usuarios
# y el modelo entrenado


def model_Logistica(x_in, RegLogist):
    x = np.asarray(x_in).reshape(1, -1)
    preds = RegLogist.predict(x)
    return preds


def model_bayes(x_in, NaBay):
    x = np.asarray(x_in).reshape(1, -1)
    preds = NaBay.predict(x)
    return preds


# Ruras donde estan guardads los modelos entrenados
Ruta1 = 'models/Regresión_Logistica.pkl'
Ruta2 = 'models/Naive_Bayes.pkl'
RegLogist = ''
NaBay = ''
with open(Ruta1, 'rb') as Lo:
    RegLogist = pickle.load(Lo)
with open(Ruta2, 'rb') as Ba:
    NaBay = pickle.load(Ba)


def main():
    st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">', unsafe_allow_html=True)
    st.markdown("""
   <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color:#FF6800;">
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disable" style="color:#FFFFFF;" href="">Predicción</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" style="color:#FFFFFF;" href="https://github.com/Leo646/IA_MachineLearning">GitHub</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" style="color:#FFFFFF;" href="https://colab.research.google.com/drive/1ZWSL4A7xfjLB_4cBEHgKN9iSW8k7EBRJ">Cuaderno de Trabajo</a>
      </li>
    </ul>
  </div>s
</nav>""", unsafe_allow_html=True)
    
    # parametros de ingreso de datos por el usirario

    def parametros_usario():
        estudio = st.slider('Tiempo de Estudio', 1, 4, 2)
        romantico = st.slider('Situación Romantica', 0, 1, 0)
        salud = st.slider('Estado de salud', 1, 5, 2)
        ausencias = st.slider('Numero de ausencias', 0, 93, 2)
        G1 = st.slider('Nota del parcial 1', 0, 20, 1)
        G2 = st.slider('Nota del parcial 2', 0, 20, 3)
        # Se almacenan los parametros en un diccionario
        datos = {'T. estudio': estudio,
                 'Situación Romantico': romantico,
                 'Salud': salud,
                 '# ausencias': ausencias,
                 'Parcial 1': G1,
                 'Parcial 2': G2
                 }

        parametros = pd.DataFrame(datos, index=[0])
        # se retornar los prametros
        return parametros
    # se llaman y visualizan los parametros
    Opciones_modelos = ['Regresion Logística', 'Naive Bayes', 'Resumen']
    modelos = st.sidebar.selectbox(
        'Elija el modelo con el que quiere predecir', Opciones_modelos)
    st.subheader(modelos)
    
    st.sidebar.markdown(''' 
                        
                        ### Rendimiento del estudiante
                        Con el siguiente proyecto se quiere predecir si 
                        el estudiante aprueba o reprueba el parcial final,
                        tomando en cuenta ciertas características del entorno
                        que le pueden afectar. Para hacer la predicción se
                        entrenaron los modelos con los datos  que abordan 
                        el rendimiento de los estudiantes en la educación 
                        secundaria de dos escuelas portuguesas. El dataset 
                        se lo puede encontrar en el siguiente enlace: 
                        https://archive.ics.uci.edu/ml/datasets/Student+Performance]
                        
                        Las variables que se usan en el proyecto tienen el 
                        siguiente significado:
                            
                            Tiempo de estudio semanal: 
                                1 -> menor a 2 horas, 
                                2 -> entre 2 a 5 horas, 
                                3 -> entre 5 a 10 horas,
                                4 -> más de 10 horas)
                            Con una relación sentimental:
                                1-> Si
                                2-> No
                                
                            Estado de salud actual:
                                1 -> muy malo 
                                5 -> muy bueno
                            Número de ausencias escolares: 
                                De 0 a 93
                            G1 (nota del primer periodo):s
                                De 0 a 20
                            G2 (nota del segundo periodo): 
                                DE 0 a 20
                        
                        
                        ''', unsafe_allow_html=True)

    # datos escogidos por el Usario
    #st.subheader('Datos escogidos por el usuario')
   
    # Opciones para escoger el modelo
   
   
    if modelos !='Resumen':
        df = parametros_usario()
        st.title('Modelos para predecir la probabilidad de aprobar el último parcial' +
                 ' de los estudiantes')
        #st.sidebar.header('Proyecto de Machine Learning')
        
        st.write(df)
    # boton la predicción
        if st.button('Predecir'):
        # Si se escoge la Regesion Logistica se presentara los datos de la prediccion
            if modelos == 'Regresion Logística':
                predictS = model_Logistica(df, RegLogist)
                #st.success('EL ESTUDIANTE: {}'.format(predictS[0]).upper())
                prediccion = predictS[0]
                x_i = np.asarray(df).reshape(1, -1)
                # si el valor es 1 aprueba
                if prediccion == 1:
                   
                    probabilidad = RegLogist.predict_proba(x_i)
                    # imprime probabilidad deaprobar
                    st.success('La Probabilidad de aprobar es : :{}'.format(
                    probabilidad[:, 1]*100))
                    st.success('El Estudiante: APRUEBA')
                else:
                    
                    probabilidad = RegLogist.predict_proba(x_i)
                    st.success('La Probabilidad de aprobar es :{}'.format(
                    probabilidad[:, 1]*100))
                    st.success('El Estudiante: REPRUEBA')
            else:
            # Si se escoge Naive Bayes se presentara los datos de la prediccion
                predictS1 = model_bayes(df, NaBay)
                #st.success('EL ESTUDIANTE: {}'.format(predictS[0]).upper())
                prediccion = predictS1[0]
                x_i = np.asarray(df).reshape(1, -1)
                if prediccion == 1:
                    
                    probabilidad = NaBay.predict_proba(x_i)
                    st.success('La Probabilidad de aprobar es :{}'.format(
                    probabilidad[:, 1]*100))
                    st.success('El Estudiante: APRUEBA')
                else:
                    
                    probabilidad = NaBay.predict_proba(x_i)
                    st.success('La Probabilidad de aprobar es :{}'.format(
                    probabilidad[:, 1]*100))
                    st.success('El Estudiante: REPRUEBA')
    
    
    else:
        st.title('Análisis exploratorio de datos')
        st.write('Conjunto de datos extraido del repositorio UCI')
        url='https://raw.githubusercontent.com/Leo646/IA_MachineLearning/master/Dataset/student-mat.csv'
        datos = pd.read_csv(url)
        st.write(datos)
        st.write('Se reduce el Dataset con las variables necesarias que pueden influir más para aprobar \n '+
                 "o supender el último parcial")
        dat= datos.drop(['famsup','famrel','school','guardian','age','address','famsize','schoolsup','Medu','Fedu','Mjob','Fjob','reason','traveltime','Pstatus','sex','activities','nursery','higher','internet','goout','freetime','Dalc','Walc', 'paid', 'failures'], axis=1)
        st.write(dat)
        st.write('Número de Filas x Columnas que tiene el Dataset')
        st.write((dat.shape))
        fig =plt.figure(figsize=(9,7))
        sb.histplot(data=dat['G3'])
        st.title('gráficos')
        plt.ylabel('número de estudiantes')
        plt.title('Notas del tercer parcial')
        st.pyplot(fig)
        st.write('Tranformacion de las variables "Romantic" y "G3" en valores binarios')
        st.info("Nota:\n"+
                "       es necesario que todas la varibles sean numéricas para que no se produzcan errores \n"+
                "       durante el entrenamiento de  los datos con los algoritmos")
        dat['G3'] = datos['G3'].apply(lambda x: 0 if x < 10 else 1)
        dat['G3'].unique()
        dat['romantic'] = datos['romantic'].apply(lambda x: 0 if x=='no' else 1)
        dat['romantic'].unique()
        st.write(dat)
        st.write('Estudiantes aprobados y reprobados del tercer parcial')
        labels = 'Aprobados', 'Reprobados'
        explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        fig1, ax = plt.subplots(figsize=(3,3))
        ax.pie(dat['G3'].value_counts(), explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=10)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1)
        
        st.write('Situación romántica del estudiante')
        labels = 'Soltero', 'En una relación'
        explode = (0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        fig1, ax = plt.subplots(figsize=(3,3))
        ax.pie(dat['romantic'].value_counts(), explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=10)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig1)
        st.write('Cantidad de estudiantes con o sin situacion romántica')
        n=dat.groupby("romantic").agg(frequency=("romantic", "count"))
        st.write(n)
        
        fig =plt.figure(figsize=(9,7))
        sb.histplot(data=dat['G2'])
        plt.ylabel('número de estudiantes')
        plt.title('Notas del segundo parcial')
        st.pyplot(fig)
        fig =plt.figure(figsize=(9,7))
        sb.histplot(data=dat['G1'])
        plt.ylabel('número de estudiantes')
        plt.title('Notas del primer parcial')
        st.pyplot(fig)
        
        
        st.title('Machine Learning')
        st.title('Separación de las variables dependientes e independientes')
        st.write('A contiuación se realizará paso a paso el proceso para'+ 
                 ' entrenar los datos mediante dos algoritmos')
        st.text("Librerias")
        st.code("from sklearn.metrics import confusion_matrix\n"+
                "from sklearn.metrics import classification_report\n"+
                "from sklearn.model_selection import train_test_split")
        
        #Se obtienen las variables independientes
        st.code("#Se obtienen las variables independientes:\n"+ 
                "X = datos.drop(['G3']\n"+
                "X.head()")
        X = dat.drop(['G3'], axis=1)
        st.write(X)
        st.code("#Se obtienen las variables depedientes:\n"+ 
              "Y = datos.drop(['G3']\n"+
              "Y.head()")
        Y=dat.pop('G3')
        st.write(Y)
        st.write("Es necesario comprender, que lo que se quiere predecir, es si los estudiantes \n"+
                "aprueban o no el último parcial. Por tal motivo se transformó la varible 'G3' en \n "+
                "valores binarios, de tal forma que el número 1 significa 'aprobado' y el número 0 'reprobado'.")

        st.code("# Se separan los datos para ajuste y prueba:\n"+ 
              "from sklearn.model_selection import train_test_split \n"+
              "X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0) \n"+
              "print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0],X_test.shape[0]))")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
        st.write('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0],X_test.shape[0]))
 
        st.header('Algoritmo de Regresión Logística')
        st.text('Desarrollada por David Cox en 1958, es un método de regresión que permite estimar \n'+ 
                'la probabilidad de una variable cualitativa binaria en función de una variable \n' +
                'cuantitativa. Una de las principales aplicaciones de la regresión logística es la \n'+
                'de clasificación binaria, en el que las observaciones se clasifican en un grupo y \n'+ 
                'otro dependiendo del valor que tome la variable empleada como predictor. \n'+

                'Ventajas \n'+

                    '   * Rara vez existe sobreajuste \n'+
                    '   * El uso de la regularización es efectivo en la selección de funciones.\n'+
                    '   * Rápido para entrenar. \n'+
                    '   * Fácil de entrenar sobre grandes datos gracias a su versión estocástica.\n'+
                    '   * Fácil de entender y explicar \n'
                'Desventajas \n'+
            
                    '   * Tienes que trabajar duro para que se ajuste a los datos no lineales. \n'+
                    '   * Puede sufrir con valores atípicos.\n'+
                    '   * En algunas ocasiones es muy simple para captar relaciones complejas entre variables.')                    
                            
                                
        st.code("# Regresión Logística \n"+
                "from warnings import simplefilter \n"+
                "simplefilter(action='ignore', category=FutureWarning)\n"+
                "from sklearn.linear_model import LogisticRegression \n"+
                "classifier = LogisticRegression()# se define el mdelo \n"+
                "classifier.fit(X_train, y_train) # se entrena el modelo \n"+

                "y_pred = classifier.predict(X_test) # se realiza laa prediiccion\n"+
                "print(Y.head(20))"+
                "print((y_pred)[0:20]) \n") 
        st.text("Comparación entre los valores verdaderos 'Y'y los valores de la predicción 'y_pred'")
        from warnings import simplefilter
        simplefilter(action='ignore', category=FutureWarning)
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression()# se define el mdelo
        classifier.fit(X_train, y_train) # se entrena el modelo
       
        y_pred = classifier.predict(X_test) # se realiza una prediiccion
        st.write(Y.head(20))
        st.write((y_pred)[0:20])
        st.subheader('Matriz de confusión')
        st.text("Es una herramienta que permite la visualización del desempeño de un algoritmo que se emplea \n"+
                "en aprendizaje supervisado."
                )
        imagen = Image.open('imagenes/matriz.PNG')
        st.image(imagen, caption='Matriz de confusión')
        st.text("Verdadero positivo: El valor real es positivo y la prueba predijo tambien que era positivo.\n"+
                "                    O bien una persona está enferma y la prueba así lo demuestra. \n"+ 
                "Verdadero negativo: El valor real es negativo y la prueba predijo tambien que el resultado \n"+ 
                "                    era negativo. O bien la persona no está enferma y la prueba así lo \n"+ 
                "                    demuestra.\n"+
                "Falso negativo: El valor real es positivo, y la prueba predijo que el resultado \n"+
                "                es negativo. La persona está enferma, pero la prueba dice de manera \n"+
                "                incorrecta que no lo está. Esto es lo que en estadística se conoce \n"+
                "                como error tipo II \n "+
                "Falso positivo: El valor real es negativo, y la prueba predijo que el resultado es \n"+
                "                positivo. La persona no está enferma, pero la prueba nos dice de \n"+ 
                "                manera incorrecta que silo está.")
        
        
        st.code("# Impresión de la matriz de confusión \n"+
                "print(confusion_matrix(y_test, y_pred))")
        st.write(confusion_matrix(y_test, y_pred))
        st.code(" from sklearn.metrics import precision_score \n"+
                " print('la presición del modelo es de :',precision_score(y_pred,y_test))")
        from sklearn.metrics import precision_score
        st.write('la presición del modelo es de :',precision_score(y_pred,y_test))
        st.text("Para cada valor de la predicción se puede calcular la probabilidad de que un estudiante\n"+
                "apruebe o repruebe el último parcial")
        st.code("st.write(y_pred)\n"+
                "a=classifier.predict_proba(X_train)\n"+
                "st.write(a)")
        st.write(y_pred)
        a=classifier.predict_proba(X_train)
        st.write(a)
        
        st.text("Entrenado el modelo se puede hacer una predicción para un estudiante.\n"+
                 "A continuación,en  la primera linea de código se agregan valores numéricos \n"+
                 "a un arreglo.  Cada valor representa lo siguiente: \n"+ 
                 "'Tiempo de estudio',\n"+
                 "'Sistuación romántica',\n"+
                 "'Salud',\n"+
                 "'Ausencias',\n"+
                 "'Notas del parcial 1(G1)',\n"+
                 "'Notas del parcial 2 (G2)'")
        st.code("x_in=np.asarray([2,1,5,10,10,10]).reshape(1,-1)\n"+
                "predict=classifier.predict(x_in)\n"+
                "print(predict[0])"
                )
        x_in=np.asarray([2,1,5,10,10,10]).reshape(1,-1)
        predict=classifier.predict(x_in)
        st.text("Con estos valores el modelo predice que el estudiante aprueba el parcial")
        st.write(predict[0])
        st.text('Para probar que la predicción del modelo es correcta, se puede calcular la probabilidad que tiene el\n'
           'estudiante de aprobar el parcial')
        st.code("a=classifier.predict_proba(x_in)\n"+
                "print(a)")
        a=classifier.predict_proba(x_in)
        st.write(a)
        st.text("Como se puede obserbar se presentan dos valores, 0.2175 (Pobabilidad de reprobar)\n"+
                "y 0.7825 (Probabilidad de aprobar). Si se observa bien, la probabilidad de aprobar\n"+
                "es mayor a 0,5 por lo que el modelo esta prediciendo correctamente")
        st.write("Se guarda el modelo")
        st.code("import pickle\n"+
                "pkl_filename='Regresión_Logistica.pkl'\n"+
                "with open(pkl_filename, 'wb')as file:\n"+
                "   pickle.dump(classifier,file)")
        st.header('Naive Bayes')
        st.text('Naive Bayes o el Ingenuo Bayes es uno de los algoritmos más simples y poderosos  \n'+ 
                'para la clasificación basado en el Teorema de Bayes con una suposición de \n' +
                'independencia entre los predictores. Naive Bayes es fácil de construir y \n'+
                'particularmente útil para conjuntos de datos muy grandes.  \n'+ 
                
                'Ventajas \n'+

                    '   * Es fácil y rápido predecir la clase de conjunto de datos de prueba. También  \n'+
                    '     funciona bien en la predicción multiclase. \n'+
                    '   * Cuando se mantiene la suposición de independencia, un clasificador Naive Bayes\n'+ 
                    '     funciona mejor en comparación con otros modelos como la Regresión Logística y se\n'+ 
                    '     necesitan menos datos de entrenamiento.\n'+
                    '   * Funciona bien en el caso de variables de entrada categóricas comparada\n'+ 
                    '     con variables numéricas. \n'+
                'Desventajas \n'+
            
                    '   * Si la variable categórica tiene una categoría en el conjunto de datos de prueba,\n'+
                    '     que no se observó en el conjunto de datos de entrenamiento, el modelo asignará \n'+
                    '     una probabilidad de 0 y no podrá hacer una predicción. Esto se conoce a menudo \n'+
                    '     como frecuencia cero. Para resolver esto, podemos utilizar la técnica de \n'+
                    '     alisamiento. Otra limitación de Naive Bayes es la asunción de predictores \n'+
                    '     independientes. En la vida real, es casi imposible que obtengamos un conjunto \n'+
                    '     de predictores que sean completamente independientes. Naive Bayes es el algoritmo \n'+
                    '     más sencillo y potente. A pesar de los significativos avances de Machine Learning\n'+
                    '     en los últimos años, ha demostrado su valía. Se ha implementado con éxito en \n'+
                    '     muchas aplicaciones, desde el análisis de texto hasta los motores de recomendación.'
                    ) 
        
        
        st.code("from warnings import simplefilter\n"+
                "simplefilter(action='ignore', category=FutureWarning)\n"+
                "from sklearn.svm import SVC\n"+

                "bayes = SVC(probability=True)\n"+
                "bayes.fit(X_train, y_train)\n"+

                "y_pred = bayes.predict(X_test)\n"+

                "# Resumen de las predicciones hechas por el clasificador\n"+
                "#print(classification_report(y_test, y_pred))\n"+
                "print(confusion_matrix(y_test, y_pred))\n"+
                "#Precision del modelo\n"+
                "from sklearn.metrics import precision_score\n"+
                "print('la presición del modelo es de :',precision_score(y_pred,y_test))\n"
                )
        
        from warnings import simplefilter
        simplefilter(action='ignore', category=FutureWarning)
        from sklearn.svm import SVC

        bayes = SVC(probability=True)
        bayes.fit(X_train, y_train)

        y_pred = bayes.predict(X_test)

        # Resumen de las predicciones hechas por el clasificador
        #print(classification_report(y_test, y_pred))
        st.write(confusion_matrix(y_test, y_pred))
        #Precision del modelo
        from sklearn.metrics import precision_score
        st.write('la presición del modelo es de :',precision_score(y_pred,y_test))

        st.code("x_in=np.asarray([2,1,5,10,10,10]).reshape(1,-1)\n"+
                "predict=bayes.predict(x_in)\n"+
                "print(predict[0])\n"+
                "b=bayes.predict_proba(x_in)\n"+
                        "print(b)")
        x_in=np.asarray([2,1,5,10,10,10]).reshape(1,-1)
        predict=bayes.predict(x_in)
        st.write(predict[0])
        b=bayes.predict_proba(x_in)
        st.write(b)
        st.text("Si se observa bien, la probabilidad de aprobar\n"+
                "es mayor a 0,5 por lo que el modelo esta prediciendo correctamente")
        st.write("Para guaradar el modelo se usa el siguinete comando ")
        st.code("import pickle\n"+
                "pkl_filename='Naive_Bayes.pkl'\n"+
                "with open(pkl_filename, 'wb')as file:\n"+
                "   pickle.dump(classifier,file)")
if __name__ == '__main__':
    main()
