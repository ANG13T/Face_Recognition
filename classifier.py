import numpy as np
from PIL import Image
import os, cv2

def train_classifier(data_dir):
    path = [os.join(data_dir, f) for f in os.listdir(data_dir)] #stores all images file name in list 
    faces = []
    ids = []

    for image in path: 
        img = Image.open(image).convert('L') #converts image to grayscale
        imageNp = np.array(img, 'uint8') #converts image data to numpy format
        id = int(os.path.split(image)[1].split('.')[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)
    clf = cv2.face.LBPHFaceRecognizer_create() #makes a classifier
    clf.train(faces, ids) #makes points with faces and labels it using ids
    clf.write("classifier.yml")

train_classifier("data")
