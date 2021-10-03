default face detector (S3FD)
import face_alignment
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('acazlolrpz.mp4')
frames = []
while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
   
 
 fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='sfd')
 
 import time
t_start = time.time()
det = fa.get_landmarks_from_image(frames[0])
print(f'SFD: Execution time for a single image: {time.time() - t_start}')


SFD: Execution time for a single image: 23.135814905166626
plt.imshow(frames[0])
for detection in det:
    plt.scatter(detection[:,0], detection[:,1], 2)

batch = np.stack(frames)
batch = batch.transpose(0, 3, 1, 2)
batch = torch.Tensor(batch[:2])
t_start = time.time()
preds = fa.get_landmarks_from_batch(batch)
print(f'SFD: Execution time for a batch of 2 images: {time.time() - t_start}')


SFD: Execution time for a batch of 2 images: 45.840161085128784
fig = plt.figure(figsize=(10, 5))
for i, pred in enumerate(preds):
    plt.subplot(1, 2, i + 1)
    plt.imshow(frames[1])
    plt.title(f'frame[{i}]')
    for detection in pred:
        plt.scatter(detection[:,0], detection[:,1], 2)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='blazeface')

t_start = time.time()
preds = fa.get_landmarks_from_image(frames[0])
print(f'BlazeFace: Execution time for a single image: {time.time() - t_start}')


BlazeFace: Execution time for a single image: 1.6376028060913086

plt.imshow(frames[0])
for detection in preds:
    plt.scatter(detection[:,0], detection[:,1], 2)
    
batch = np.stack(frames)
batch = batch.transpose(0, 3, 1, 2)
batch = torch.Tensor(batch[:2])
t_start = time.time()
preds = fa.get_landmarks_from_batch(batch)
print(f'BlazeFace: Execution time for a batch of 2 images: {time.time() - t_start}')
BlazeFace: Execution time for a batch of 2 images: 3.1170198917388916

fig = plt.figure(figsize=(10, 25))

for i, pred in enumerate(preds):
    plt.subplot(5, 2, i + 1)
    plt.imshow(frames[i])
    plt.title(f'frame[{i}]')
    for detection in pred:
        plt.scatter(detection[:,0], detection[:,1], 2)

