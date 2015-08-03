# faceTimer
Automatic tagging of videos

# Problem

The objective is to rank the actors in a given video, in decreasing order of their screen time.

Successful recognition can be shown by generating 6 videos of length same as the input video, each with a bounding box for a
particular actor’s face.

# Proposed approach

- Run a video summarization algorithm or keyframe detection algorithm to reduce the video data.
- Run a facedetector on each frame to get the faces that appear in the whole video.
- Cluster the faces
- Arrange in decreasing order of populated clusters

# WIP - Solution

(Download http://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip)

To solve the assignment, I divided the project into three logical units:
- Capturing faces (Face Detection).
- Clustering faces.
- Tagging faces (Face Recognition).

The critical unit according to me is, how we cluster the faces. I have tried to solve the clustering problem using two approaches:

- Supervised Learning - using openCV’s fisherFaceRecognizer
- Unsupervised Learning - using K-means clustering

As I found out, the supervised learning approach produced more reliable data than unsupervised learning.  
There are 2 python programs:
- solution.py (Supervised Learning method that addresses all the requirements)
- misc.py (Unsupervised k-means clustering method that only tries to find the top-6 actors)

# How to run the programs
1. To run the program, the user will run it as:  
python solution.py fileName.ext summarizeFactor  
The ”summarizeFactor” argument will decide how many frames must be skipped. I have not implemented any existing keyframe detection or video summarization algorithms. But this argument can help to achieve the goal.  
The factor is multiplied by 24. Eg. If the user specifies the factor as 1, then only 1 frame is fetched out of every 24*1=24 frames (so approx. 1 frame/sec). If 0.5, then 2 frames/ 24 frames are analyzed. If 0, then every frame is read.

2. Firstly, I train a model using openCV’s createFisherFaceRecognizer() method. The training set that I use is the face database compilation by AT&T.

3. After I have trained the model, I capture the video stream of the input video file.  
Now, my objective is to detect faces in this video stream. To do this, I use ”haarcascade frontalface” cascade file with openCV’s cascade classifier.
Using the arguments scaleFactor and minNeighbors in the function detectMultiScale(), I have tried to optimize face detection.  
For each face that I detect, I run a prediction on it using the model that I have previously created. This classifies the face into 1 out of the 40 classes that AT&T’s database has.

4. I create a dictionary that has each of the 40 labels as the keys. As and when a face is classified into a label, that face is mapped to the dictionary’s key. So every key has a list of frames as the value.  
Once all the frames are analyzed and the faces captured, I sort the the dictionary in descending order depending on the number of faces in the list of each label. The top six labels are the required actors.

5. I create a directory called ”faceData” that has 6 sub-directories - 0,1,2,3,4,5.  
The directory ”0” has faces of the actor with maximum screentime, and so on, (in descending order) uptil directory 5.  
(My solution is limited to showing all the faces captured for an actor. This is because many predictions include false-positives)

6. Now, I try to tackle the second requirement of the task - Tagging actors in the videos.  
To do this, I again train a model using createFisherFaceRecognizer() but this time providing our captured faces as input.  
Each frame of the video is analyzed and six frames are generated. Each of the six frames, tries to tag its respective actor. I finally generate six
videos using openCV’s VideoWriter.  
(Please note that this can be a very long process depending on the size of the input video)

# Alternate Solution
(Unsupervised k-means clustering)

1. In this approach, I do not train any model and just directly capture all the faces in the video stream and store them in a list.

2. Next, I try to cluster the faces using the k-means clustering algorithm.  
I found it easier to use ”cluster” from ”sklearn” module rather than openCV’s k-means clustering.  
In this program, I first use openCV’s ORB feature detector to detect key-points and descriptors for each face. I use these descriptors as features and
feed them to sklearn’s clustering method for k-means. Also, the program specifies that there will be 6 clusters.

3. In the program, I maintain a dictionary to keep a track of the mapping for a frame to a label. Using this mapping, I write faces (in a similar
fashion as above) into the faceData directory’s sub-directories 0,1,2,3,4,5 (in decreasing order).

4. However, using this method I could not generate an acceptable result. I have included this work in the misc.py file. It can be run as:  
python misc.py fileName.ext

# Results
On running the program solution.py, the user is first asked if they already have the faces stored in the faceData directory. This helps in the case, when the user has already run the program once or has faces of the actors, already available as training data. However, to find out the respective screen-time, this module has to be run. It has to be kept in mind, that the summarizeFactor may skew the result.

On choosing to generate faces, faceData directory and 6 sub-directories are created, that have faces of actors in decreasing order of screen-time. However, there were a lot of false-positives and false-negatives in the predictions. After storing faces or if the user has chosen not to collect faces, the program asks the user if they want to go ahead and generate the videos? (At this point the user may want to remove bad data from these folders for better tagging.)

The tagged videos are stored in the ”videoData” directory.

# References

1. https://realpython.com/blog/python/face-recognition-with-python/
2. http://opencvpython.blogspot.in/2013/01/k-means-clustering-3-working-with-opencv.html
3. http://noahingham.com/run/blog?q=facerec-python-opencv
4. http://www.tp.umu.se/nylen/pylect/advanced/scikit-learn/index.html
