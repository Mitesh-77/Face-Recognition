import tkinter as tk
from tkinter import Message, Text, messagebox
import cv2, os, operator
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

path = "UnknownImages"

root = tk.Tk()

root.geometry("800x500")
root.title("Face Recogniser")
root.resizable(False,False)
#root.configure(bg="#0a0352")
filename = tk.PhotoImage(file = "E:\\mitesh\\Tkinter\\background.png")
background_label = tk.Label(root, image=filename)
background_label.place(x=0, y=0, relwidth=1, relheight=1)



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def takeImages():
    Student_Id = Stu_id.get()
    Student_name = Stu_name.get()
    if(is_number(Student_Id) and Student_name.isalpha()):
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        cam = cv2.VideoCapture(0)
        sampleNum = 0
        while(True):
            success, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.2, 5)
            for (x, y, w, h) in faces:
                # (x,y) = startpoint for box
                # (x+w,y+h) = endpoint for box
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImages
                dir = "E:\mitesh\Tkinter\TrainingImages"
                dir = os.path.join(dir,f"{Student_name}_{Student_Id}")
                if not os.path.exists(dir):
                    os.makedirs(dir)
                cv2.imwrite(f"{dir}\ " + Student_name + "_" + Student_Id + '_' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('webcam', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 60
            elif sampleNum > 60:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Student_Id + " Name : " + Student_name
        row = [Student_Id, Student_name]
        Columns = ['Id','Name']
        data = pd.read_csv("StudentDetails\StudentDetails.csv",names=Columns)
        ids = data.Id.tolist()
        if row[0] not in ids:
            with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
        data = pd.read_csv("StudentDetails\StudentDetails.csv")
        data = data.sort_values(by=["Id"])
        data.to_csv("StudentDetails\StudentDetails.csv", index=False)
        messagebox.showinfo("showinfo", f"{res}")
    else:
        if (is_number(Student_Id)):
            res = "Enter Alphabetical Name"
            messagebox.showinfo("showinfo", f"{res}")
        elif (Student_name.isalpha()):
            res = "Enter Numeric Id"
            messagebox.showinfo("showinfo", f"{res}")
        else:
            res="Enter Some Value for Name and Id"
            messagebox.showinfo("showinfo", f"{res}")


def getAttendace_csv(path, tms):
    Time = 0
    Attendance_csv = []
    Attendance_csv.extend([os.path.join(path, f) for f in os.scandir(path)])
    #print(Attendance_csv)
    #print(tms)
    #create time List
    if len(Attendance_csv)!= 0 :
        for csvfile in Attendance_csv:
            #os.path.split(csvfile) ---->('E:\\mitesh\\Tkinter\\Attendance\\ML\\2021-02-01', 'Attendance_ML_2021-02-01_11.csv')
            hour = int(os.path.split(csvfile)[-1].split(".")[0].split("_")[-1].split("-")[0])
            minute = int(os.path.split(csvfile)[-1].split(".")[0].split("_")[-1].split("-")[1])
            total_minute =  (hour * 60) + minute
            #print(total_minute)
            if tms - total_minute<60:
                Time=total_minute
            else :
                Time = tms
        #print(tms,total_minute,Time)
    return Time


def getImagesAndLabels(path):
    imagePaths = []
    for item in os.listdir(path):
        #print(item)
        path1 = os.path.join(path,item)
        #print(path1)
        # get the path of all the files in the folder
        imagePaths.extend([os.path.join(path1, f) for f in os.listdir(path1)])
    #print(imagePaths)


    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        #print(os.path.split(imagePath))  --> ('TrainingImages', ' Mitesh_104_1.jpg')
        Id = int(os.path.split(imagePath)[-1].split("_")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


def trainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImages")
    recognizer.train(faces, np.array(Id))
    recognizer.save("Images_label\Trainner.yml")
    set_id = set(Id)
    res = "Image Trained for id " + ",".join(str(f) for f in set_id)
    messagebox.showinfo("showinfo", f"{res}")


def trackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("Images_label\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns=col_names)
    while True:
        Success, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):
                ts = time.time()
                #print(ts)
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                #print(date,timeStamp)
                aa = df[df['Id']==Id]['Name'].values
                tt = str(Id) + "-" + aa
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
            else:
                Id = 'Unknown'
                tt = str(Id)
            if (conf > 75):
                noOfFile = len(os.listdir("UnknownImages")) + 1
                cv2.imwrite("UnknownImages\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    sub = "ML"
    dir = "E:\mitesh\Tkinter\Attendance"
    dir = os.path.join(dir, f"{sub}")
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.join(dir, f"{date}")
    if not os.path.exists(dir):
        os.makedirs(dir)

    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    total_minits = (60 * int(Hour)) + int(Minute)
    previous_csv_minute = getAttendace_csv(dir, total_minits)
    #print(previous_csv_minute,total_minits)
    Attendancecsv = []
    Attendancecsv.extend(os.path.join(dir, f) for f in os.scandir(dir))
    if previous_csv_minute != total_minits and len(Attendancecsv)!=0:
        pch = int(previous_csv_minute/60)
        #print(Attendancecsv)
        for f in Attendancecsv:
            if int(os.path.split(f)[-1].split(".")[0].split("_")[-1].split("-")[0])==pch:
                fileName=f"{dir}\{os.path.split(f)[-1]}"
                filerow = pd.read_csv(fileName)
                attendance = attendance.append(filerow,ignore_index=True)
                attendance.sort_values("Id", inplace=True)
                attendance.drop_duplicates(subset="Id", keep="first", inplace=True)
                #print(attendance.append(filerow,ignore_index=True))
    else :
        fileName = f"{dir}\Attendance-{sub}_{date}_{Hour}-{Minute}-{Second}" + ".csv"
    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()
    #print(attendance)+
    res = attendance
    eAttandance.configure(text=res)


# tk.Label(root, text="Face Recognition Based Attendance System", fg="white",
#                 bg="#0a0352",pady=20, font="Times 25 bold underline", underline=5).place(x=100, y=10)
#
# name = tk.Label(root, text="Enter Name : ", fg="white", bg="#0a0352", font="verdana 15")
# id = tk.Label(root, text="Enter Id       : ", fg="white", bg="#0a0352", font="verdana 15")
#notification = tk.Label(root, text="Notification  : ", fg="white", bg="#0a0352", font="verdana 15")

# name.place(x=150, y=100)
# id.place(x=150, y=150)
#notification.place(x=150, y=200)


Stu_name = tk.Entry(root, width=30, bg="white",  font="Times")
Stu_id = tk.Entry(root, width=30, bg="white",  font="Times")
#enotification = tk.Label(root, text="" , width=45, bg="white", font="Times")

Stu_name.place(x=310, y=105)
Stu_id.place(x=310, y=155)
#enotification.place(x=300, y=205)

frame = tk.Frame(root, borderwidth=3, bg="white", relief="sunken").pack()

b1 = tk.Button(frame, text="Take Photo", height=2, width=15, cursor="hand2", font="Arial 13 bold", command=takeImages)
b1.place(x=150, y=270)

b2 = tk.Button(frame, text="Train Photo", height=2, width=15, cursor="hand2",  font="Arial 13 bold", command=trainImages)
b2.place(x=350, y=270)

b3 = tk.Button(frame, text="Test Photo", height=2, width=15,  cursor="hand2", font="Arial 13 bold", command=trackImages)
b3.place(x=550, y=270)

# Attendance = tk.Label(root, text="Attendance  : ", fg="white", bg="#0a0352", font="verdana 15")
# Attendance.place(x=150, y=350)

eAttandance = tk.Label(root, text="", width=51, bg="white", font="Times")
eAttandance.place(x=305, y=355, height=40)

b4 = tk.Button(root, text="Quit", height=2, width=15, cursor="hand2",  font="Arial 13 bold", command=root.destroy)
b4.place(x=550, y=425)


root.mainloop()
