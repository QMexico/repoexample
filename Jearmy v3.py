# Este es un comentario nuevo
import cv2, os
import imutils as imu
import pyttsx3 as Spk
import numpy as np

Jrmy_Voice = Spk.init()
voices = Jrmy_Voice.getProperty('voices')
voices = Jrmy_Voice.setProperty('voice', voices[1].id)

dataPath = 'C:/Users/pablo/PycharmProjects/pythonProject4/Faces_Folder'
hardCas = 'haarcascade_frontalface_default.xml'
FaceFile_1 = 'Face_Model_B.xml'

def Spk_Sen(Message):
    Jrmy_Voice.say(Message)
    Jrmy_Voice.runAndWait()

def PersonRegis(PrsnNm, Count_int, Count_End, VideoCap_Or):
    personName = PrsnNm
    personPath = dataPath + '/' + personName

    if not os.path.exists(personPath):
        Spk_Sen("Nuevo rostro detectado")
        os.makedirs(personPath)
    cap = cv2.VideoCapture(VideoCap_Or)
    faceClss = cv2.CascadeClassifier(cv2.data.haarcascades + hardCas)
    count = Count_int

    while True:
        ret, frame = cap.read()
        if ret == False: break
        frame = imu.resize(frame, width= 640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxframe = gray.copy()

        faces = faceClss.detectMultiScale(gray, 1.2, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, h), (0, 255, 0), 2)
            rostro = auxframe[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/Face_No_{}.jpg'.format(count), rostro)
            count += 1
        cv2.imshow('Face_Capture',gray)

        k = cv2.waitKey(1)
        if k == 27 or count >= Count_End:
            break
    cap.release()
    cv2.destroyAllWindows()

def Face_Trainer():
    peoplelist = os.listdir(dataPath)

    labels = []
    facesData = []
    label = 0

    Spk_Sen("Leyendo los rostros, por favor espere")
    for NameDir in peoplelist:
        personDatPath = dataPath + '/' + NameDir

        for fileName in os.listdir(personDatPath):
            labels.append(label)
            facesData.append(cv2.imread(personDatPath + '/' + fileName, 0))
            image = cv2.imread(personDatPath + '/' + fileName, 0)
        label += 1
    Spk_Sen("Lectura terminada")

    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    Spk_Sen("Memorizando rostros")
    face_recognizer.train(facesData, np.array(labels))
    face_recognizer.write('Face_Model_B.xml')
    Spk_Sen("Rostros memorizados")

def Face_Rec(VideoCap_Or):
    imagePaths = os.listdir(dataPath)
    face_recognizer = cv2.face.EigenFaceRecognizer_create()

    face_recognizer.read(FaceFile_1)
    cap = cv2.VideoCapture(VideoCap_Or)
    face_class = cv2.CascadeClassifier(cv2.data.haarcascades + hardCas)
    Spk_Sen("Iniciando modo de visualización")
    while True:
        ret, frame = cap.read()
        if ret==False: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxframe = gray.copy()

        faces = face_class.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rostro = auxframe[y: y + h, x: x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation = cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            cv2.putText(frame, '{}'.format(result), (x, y + 20), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

            if result[1] < 8000:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]), (x, y - 25), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            else:
                cv2.putText(frame, 'Desconocido', (x, y - 25), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
