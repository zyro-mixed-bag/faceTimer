import cv2
import sys
import os
import copy
import numpy as np

"""
@Author: Chirag Agrawal

Usage : python solution.py <input_filename.extension> <video_summarize_factor>

'faceData' Directory will have 6 sub-directories. 
Dir '0' will have the faces of the actor with the longest screen time
In order of decreasing screen time -> Dir '0', '1', '2', '3', '4', '5'
These faces will later be used to train the model for face recognition for the optional output

If the user chooses to save videos wrt each face, 6 videos will be generated in 'videoData' directory
"""

faceDataDir = 'faceData'
labelFaceDic = {}
names = {}
size = 4
(im_width, im_height) = (112, 92)
try:
    videoPath = sys.argv[1]
    factor = float(sys.argv[2])*24
except IndexError:
    print "\nUsage : python solution.py <input_filename.extension> <summarize_factor>\n"
    exit(0)

    
    
def main():        
    for i in range(40):
        labelFaceDic.update({'s%s' % (i+1) : []})
    
    flag = raw_input("\nIf you already have faces stored in 'faceData' directory, you can skip capturing faces and directly generate videos.\n\nDo you want to continue capturing faces?  (y/n) : ")
    
    if flag == 'y':    
        makeDirectoryStructure()    
        trainingDir = 'att_faces'
        model = trainer(trainingDir)
        captureFaces(model)
    createVideos()
    

"""Make the faceData directory and its subdirectories in the root folder"""
def makeDirectoryStructure():
    if not os.path.isdir(faceDataDir):
        os.mkdir(faceDataDir)
    for i in range(6):
        if not os.path.isdir(faceDataDir + "/%s" % (i)):
            os.mkdir(faceDataDir + "/%s" % (i))     


"""Trains a model using the att_faces or facceData directory"""
def trainer(trainingDir):
    #Train a model using fisherFaceRecognizer
    print('\n.......Now Training Model.......')
    (images, labels, id) = ([], [], 0)
    
    for (subdirs, dirs, files) in os.walk(trainingDir):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(trainingDir, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1          
    (images, labels) = [np.array(lis) for lis in [images, labels]]
    model = cv2.createFisherFaceRecognizer()
    model.train(images, labels)
    return model


"""Capture Faces from the input video using "haarcascade_frontalface" cascade file in the root folder"""
def captureFaces(model):    
    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(videoPath)
    
    #pos_frame defines the index of the next frame to load
    pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    tot_frame = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    limit = tot_frame - factor
    counter = 0
    
    print('\n.......Now Capturing Faces (Total %s frames).......' % (tot_frame))
    #Capture video and use fisherFaceRecognizer on input video
    while pos_frame<limit:
        flag, frame = cap.read()
        
        if flag:        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
            for face_i in faces:
                (x, y, w, h) = [v * 1 for v in face_i]
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (im_width, im_height))
                prediction = model.predict(face_resize)
                
                labelFaceDic[names[prediction[0]]].append(face_resize)
                counter += 1
                sys.stdout.write("Faces captured: %d in %d frames\r" % (counter, pos_frame) )
                sys.stdout.flush()
        else:
            #The next frame is not ready, reading it again
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
            cv2.waitKey(1000)        

        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
           
        #Caputring 1 frame out of factor*24 frames for summarizing
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame+factor)
      
  
    cap.release()
    cv2.destroyAllWindows()

    print "\n\n.........Storing faces (in decreasing order of screentime)\nIn directories: faceData/0 ... faceData/5\n"
    print "\nThe result of the analysis is as follows:"
    
    for (i, label) in enumerate(sorted(labelFaceDic.iterkeys(), key=lambda k: len(labelFaceDic[k]), reverse=True)[:6]):
        print ("\nActor %s --- %s frames --- %s seconds" % (i, len(labelFaceDic[label]), int(len(labelFaceDic[label])/fps)))
        for (j, face) in enumerate(labelFaceDic[label]):
            cv2.imwrite(faceDataDir+"/%s/%s.jpg" % (i, j), face)
            

"""Creating 6 videos with each video tagged with an actor (decreasing order)"""
def createVideos():
    flag = raw_input("\n\nDo you want to generate videos with recognized faces? (y/n) : ")
    if flag == 'n':
        return
        
    if not os.path.isdir('videoData'):
        os.mkdir('videoData')
             
    model = trainer(faceDataDir)
    print("\n.......Now generating videos in 'videoData' directory.......\nThis might take a while. Press Ctrl+C to stop the process.")
    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(videoPath)
    
    # Initialize codec, frames/sec, width & height
    fourcc = cv2.cv.CV_FOURCC(*'XVID')  
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    w = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    
    #Create 6 VideoWriter objects
    video_writer = []    
    for i in range(6):        
        video_writer.append(cv2.VideoWriter("videoData/output%s.avi" % (i), fourcc, fps, (w, h)))
    
    #Use fisherRecognizer on input video
    while True:
        flag2, frame = cap.read()
        
        #Each actor will have only their tag on the frame
        actorFrames = []
        for i in range(6):
            actorFrames.append(copy.deepcopy(frame))
            
        if flag2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
            for i in range(len(faces)):
                face_i = faces[i]
                (x, y, w, h) = [v * 1 for v in face_i]
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (im_width, im_height))
                prediction = model.predict(face_resize)
                
                #Tagging faces with rectangles
                for i in range(6):
                    if (int(names[prediction[0]]) == i):
                        cv2.rectangle(actorFrames[i], (x, y), (x + w, y + h), (0, 255, 0), 3)
                        cv2.putText(actorFrames[i], '%s' % (names[prediction[0]]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            
            for i in range(6):
                video_writer[i].write(actorFrames[i])
                
        if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) >= cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            break
            
    cap.release()
    for i in range(6):
        video_writer[i].release()
    cv2.destroyAllWindows()    
        
        
if __name__=="__main__":main()
