import cv2
import numpy as np
import sys
import os
import pickle

face_classifier = cv2.CascadeClassifier(
    'resources/classifiers/haarcascade_frontalface_alt2.xml')

recognizer_path = "resources/pickle/recognizer.pickle"
embed_model = "resources/model/openface_nn4.small2.v1.t7"
embed_path = "resources/pickle/embeddings.pickle"
fisher_recognizer = cv2.face.FisherFaceRecognizer_create()

embedder = cv2.dnn.readNetFromTorch(embed_model)  # carga del modelo embedder
fisher_recognizer.read("resources/trainer_fisher.yml")

with open("resources/pickle/labels.pickle", "rb") as f:
	inv_labels = pickle.load(f)
	labels = {v: k for k, v in inv_labels.items()}
recognizer = pickle.loads(open(recognizer_path, "rb").read())

video_interface = cv2.VideoCapture(0)
n = 0
# vamos a crear datasets de por ejemplo 50 fotos :D


def image_preprocessing(img):
	# ampliamos el espectro de la imagen para normalizar la intesidad de pixel (mayor contraste, mejor extracción de features)
	return cv2.equalizeHist(img)


while True:
	# Leemos el video frame a frame
    ret, frame = video_interface.read()
    # Lo pasamos a escala de grises
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detectamos las caras
        faces = face_classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 150), 2)
            w_rn = int(-0.1*w/2)
            roi_color = frame[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x+w_rn:x+w-w_rn]

            id_, confidence= fisher_recognizer.predict(cv2.resize(roi_gray, (64,64)))
            cv2.putText(frame, "Fisher->"+labels[id_],(x+150,y-35),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,150,255),2,cv2.LINE_AA)

            font = cv2.FONT_HERSHEY_SIMPLEX
            try:
                faceBlob = cv2.dnn.blobFromImage(roi_color, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
            except:
                print("Cara no centrada en el campo de vision de cámara. Frame corrupto")

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            if proba*100>50:
                name = labels[j]  # le.classes_[j]
                
                text = "{}: {:.2f}%".format(name, proba * 100)

                cv2.putText(frame, text, (x, y-35),
                    font, 0.45, (0, 0, 255), 2)

            cv2.putText(frame, "Carita detectada :)",(x,y-5),font, 1, (255,255,255),2,cv2.LINE_AA)

			

        cv2.imshow('Video WebCam', frame) 
        # Rompemos si pretamos 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Liberamos la interfaz de video y destruimos las ventanas creadas
print('Saliendo...')
video_interface.release()
cv2.destroyAllWindows()
