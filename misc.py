import cv2
import sys
import os
import numpy as np
import itertools

from matplotlib import pyplot as plt
from sklearn import cluster, datasets
from pprint import pprint


"""Usage : python final.py <input_filename.extension>"""

"""
    'faceData' Directory will have 6 directories. 
    Dir '0' will have the faces of the actor with the longest screen time
    In order of decreasing screen time -> Dir '1', '2', '3', '4', '5'
    These faces will later be used to train the model for face recognition for the optional output
"""

imageFaces = [] 
faceDataDir = 'faceData'

def main():
    #Capture Faces using HAAR Frontal Face classifier
    captureFaces()
    #Clustering Faces
    clusterFaces()


def makeDirectoryStructure():
    if not os.path.isdir(faceDataDir):
        os.mkdir(faceDataDir)
    for i in range(6):
        if not os.path.isdir(faceDataDir + "/%s" % (i)):
            os.mkdir(faceDataDir + "/%s" % (i))     
                
def captureFaces():
    videoPath = sys.argv[1]

    #Haar Frontal face Feature
    haarFrontFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    print "\n.......Now capturing faces......."

    cap = cv2.VideoCapture(videoPath)
    
    #pos_frame defines the index of the next frame to load
    pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    
    counter = 0;    
    while True:
        flag, frame = cap.read()
        if flag:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces in the frame
            faces = haarFrontFace.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                
            # Collect grayScale faces in a list
            for (x, y, w, h) in faces:
                counter +=1
                sys.stdout.write("Faces captured: %d in %d frames\r" % (counter, pos_frame) )
                sys.stdout.flush()
                imageFaces.append(gray[y:(y+h), x:(x+w)])
        
        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        
        #Caputring 2 frames in a second. Assuming, there is not much change in a second
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame+12)
        
        #Break if ESC is pressed
        if cv2.waitKey(10) == 27:
            break
        
        # Break if the number of captured frames is equal to the total frames
        if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) >= cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
        #if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) >= 2000.0:
            break
            

def clusterFaces():    
    #desList will store descriptor for each frame
    desList = []
    
    #labeledFrames will help to keep a track of frames for which descriptors were found successfully
    labeledFrames = []
    
    print "\n\n.........Running k-means clustering on the faces......"
    #Detecting keypoints and descriptors. I use ORB since it is free to use
    orb = cv2.ORB()
    for (counter,img) in enumerate(imageFaces):
        kp = orb.detect(img,None)
        kp, des = orb.compute(img, kp)
        if des is not None:
            desList.append(np.float32(des[0]))
            labeledFrames.append(counter)

    #z will store the descriptors as a numpy array. z will have the features for k-means clustering for 6 labels
    z = np.asarray(desList)
    k_means = cluster.KMeans(6)
    k_means.fit(z)

    print "\nThe following is the result of the analysis: \n"
    #frameLabelDic will be a dictionary to map a frame to an actor
    frameLabelDic = {}
    #labelCount will give the total number of frames mapped to one actor. 6 mappings in total
    labelCount = {}

    for label in range(6):
        labelCount.update({label:0})

    for x in range(len(labeledFrames)):
        frameLabelDic.update({labeledFrames[x]:k_means.labels_[x]})
        labelCount[k_means.labels_[x]]+=1
    
    #pprint(frameLabelDic)
    pprint(labelCount)    
    
    #Now storing faces in directories
    print "\n.........Storing faces \nIn directories: faceData/0 ... faceData/5\n"

    makeDirectoryStructure()
    for faceFrame in frameLabelDic.keys():
        cv2.imwrite(faceDataDir+"/%s/%s.jpg" % (frameLabelDic[faceFrame], faceFrame), imageFaces[faceFrame])       
            
if __name__=="__main__":main()            
