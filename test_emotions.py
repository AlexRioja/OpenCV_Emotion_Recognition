from sklearn.svm import SVC
import numpy as np
import cv2
import os
import imutils
import pickle
import matplotlib.pyplot as plt

face_classifier = cv2.CascadeClassifier(
    'resources/classifiers/haarcascade_frontalface_alt2.xml')
dataset_path = "resources/dataset_ck_modded"
#cd dataset_path="resources/dataset_abelosorio_not_modded" # or "resources/dataset_abelosorio"
embed_model = "resources/model/openface_nn4.small2.v1.t7"
embed_path = "resources/pickle/embeddings.pickle"
embed_path_test = "resources/pickle/embeddingsTest.pickle"

embedder = cv2.dnn.readNetFromTorch(embed_model)  # carga del modelo embedder
fishface = cv2.face.FisherFaceRecognizer_create() 

embeddings = []
labels = []

embeddingsTest = []
labelsTest = []

images_array = []

label_ids = {}
current_id = 0


def image_preprocessing(img):
    # ampliamos el espectro de la imagen para normalizar la intesidad de pixel (mayor contraste, mejor extracción de features)
    return cv2.equalizeHist(img)


def create_ids_4_labels(label):
    global current_id
    id_ = 0
    if not label in label_ids:
        label_ids[label] = current_id
        current_id += 1
    id_ = label_ids[label]
    return id_


count = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "_").lower()
            print("Procesando :"+label+"_"+str(count))
            image = cv2.imread(path)
            image = imutils.resize(image, width=96)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            for (x, y, w, h) in faces:
                w_rn = int(-0.1*w/2)
                roi_gray = gray[y:y+h, x+w_rn:x+w-w_rn]

                faceBlob = cv2.dnn.blobFromImage(cv2.cvtColor(image_preprocessing(gray), cv2.COLOR_GRAY2BGR), 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                print(vec)
                print(vec.flatten())
                if count % 10 == 0:
                    labelsTest.append(create_ids_4_labels(label))
                    embeddingsTest.append(vec.flatten())
                else:
                    images_array.append(cv2.resize(gray,(64,64)))
                    labels.append(create_ids_4_labels(label))
                    embeddings.append(vec.flatten())
                count += 1

print(vec.shape)


print(label_ids)
print(labels)
print(len(labels))
print("Procesados "+str(count)+" archivos con éxito :)")

dataTest = {"embeddings": embeddingsTest, "names": labelsTest}
data = {"embeddings": embeddings, "names": labels}

with open(embed_path, "wb") as f:
    f.write(pickle.dumps(data))

with open(embed_path_test, "wb") as f:
    f.write(pickle.dumps(dataTest))

with open("resources/pickle/labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

print("Entrenando al señor Fisher, por favor tenga paciencia..")

fishface.train(images_array, np.array(labels))
fishface.save("resources/trainer_fisher.yml")

print("Saliendo..")
