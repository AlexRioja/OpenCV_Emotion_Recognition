from sklearn.svm import SVC 
from sklearn.multiclass import OneVsRestClassifier
import argparse
import pickle
import numpy as np


#Modificar las siguientes variables al gusto del directorio montado

embed_path="resources/pickle/embeddings.pickle"
recognizer_path="resources/pickle/recognizer.pickle"
embed_path_test = "resources/pickle/embeddingsTest.pickle"


# load the face embeddings

data = pickle.loads(open(embed_path, "rb").read())
dataTest = pickle.loads(open(embed_path_test, "rb").read())

labels = data["names"]
labelsTest = dataTest["names"]

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("Entrenando al modelo...")
#el parámetro C=1.0 sobra porque es el valor por defecto
recognizer = OneVsRestClassifier(SVC(C=1, kernel="linear", probability=True),n_jobs=-1)
#recognizer= SVC(C=1, kernel='rbf', probability=True, gamma=2)
np_labels=np.array(labels)
recognizer.fit(data["embeddings"], np_labels)

#Añadiendo testing del modelo para controlar rendimiento y precisión
accuracy=recognizer.score(dataTest["embeddings"], np.array(labelsTest))
print("Modelo: "+str(accuracy*100))



with open(recognizer_path, "wb") as f:
	f.write(pickle.dumps(recognizer))

print("Saliendo..")