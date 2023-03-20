# importing necesaary libraries
import cv2
import numpy as np
import face_recognition
import csv
import datetime
import os
import time

path = "ImagesAttendence"
images = []  # store the actual images
classNames = []  # store the name of images
# get list of all the names of images in the given path
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0]) # grabs only the name rather than names.jpg


# function of find the enocdings of all the images
# we will use these encoding to match the image coming from the webcam
def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enocde = face_recognition.face_encodings(img)[0]
        encodeList.append(enocde)
    return encodeList


# def markAttendance(name):
#     with open('Attendance.csv', 'r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             entrytime = now.strftime('%H:%M:%S')
#             f.writelines(f'{name},{entrytime}\n')


def markAttendance(filename, name):
    entry_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exit_time = ""

    # Check if the student already exists in the file
    rows = []
    with open(filename, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if row[0] == name:
                # Update the exit time if the student already entered
                row[2] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                exit_time = row[2]
            rows.append(row)

    # Add a new row for the student if they haven't entered before
    if exit_time == "":
        row = [name, entry_time, exit_time]
        rows.append(row)

    # Write the updated CSV file
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(rows)


encodeListKnown = findEncoding(images)
print('encoding complete')

# let's get the images from the webcam

cap = cv2.VideoCapture(0)
flag = True
while True:
    success, img = cap.read()
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:

            name = classNames[matchIndex].upper()
            # drawing a rectangle around the face
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            if flag:
                markAttendance("Attendance.csv",name)
                flag = False


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

# faceLoc = face_recognition.face_locations(imgElon)[0]
# encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
#
# faceLocTest = face_recognition.face_locations(imgTest)[0]
# encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)
#
# results = face_recognition.compare_faces([encodeElon], encodeTest)
# faceDis = face_recognition.face_distance([encodeElon], encodeTest)




