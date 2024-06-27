## Homework 10

In this homework, you are going to use and compare two different trackers (of your liking) and compare the results.

### Step 1
Decide what video you are going to use for this homework, select an object and generate the template. You can use any video you want (your own, from Youtube, etc.)
and track any object you want (e.g. a car, a pedestrian, etc.).

### Step 2
Initialize a tracker (e.g. KCF).

### Step 3
Run the tracker on the video and the selected object. Run the tracker for around 10-15 frames.

### Step 4
For each frame, print the bounding box on the image and save it.

### Step 5
Select a different tracker (e.g. CSRT) and repeat steps 2, 3 and 4.

### Step 6
Compare the results:
* Do you see any differences? If so, what are they?
* Does one tracker perform better than the other? In what way?


## Results
For this homework I tried to track faces of people. There are two python files in this homework [text](homework-only-tracking.py), where I used only a tracker and [text](homework-tracking-with-detection.py), where I used a combination of face_detection_yunet detector and tracking. I also compared KCF and CSRT trackers.
I found the following:
* CSRT tracker is much slower compared to KCF, but it scales the bounding box when the object size changes, it is also more accurate and manages to not lose the object for a longer time.
* KCF tracker is much faster, but the bounding box is always of the same size, and the accuracy is lower.
* When you use a detector with the combination of tracker, you can get better results. Although in case of face_detection_yunet, on the test video I had, I found that this detector does not detect small faces in the distance well.
