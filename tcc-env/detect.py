#!/usr/bin/env python
# coding: utf-8

# In[12]:


import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model
import time
import ipynb.fs.full.crop_eye_func as cef
from pygame import mixer


# In[13]:


# configurar reprodutor de aúdio
mixer.init()
mixer.music.load('alarme.wav')
mixer.music.set_volume(1)

# tamanho da janela de apresentação
im_size = (34, 26)

# contar tempo que os olhos permanecem fechados
eye_timer = 0

# variáveis auxiliares de cálculo de fps
prev_frame_time = 0
new_frame_time = 0

# hog - detecção de face
detector = dlib.get_frontal_face_detector()
# landmark detection - detecção ocular
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# modelo de classificação ocular
model = load_model('models/2022_01_09_11_02_24.h5')
model.summary()


# In[14]:


# main
# inicialização da câmera
cap = cv2.VideoCapture(0)
ret,img = cap.read()
i = 0

while ret:
    # frame
    ret, img = cap.read()
    img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
    
    # imagem em escala cinza para melhorar o processamento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # cálculo fps
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    fps = "{:.2f}".format(fps)
    prev_frame_time = new_frame_time
    fps = str(fps)
 
    # apresentação do fps na tela
    cv2.putText(img, "FPS:" + fps, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    faces = detector(gray)
    
    for face in faces:
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)

        eye_img_l, eye_rect_l = cef.crop_eye(img=img, gray=gray, eye_points=shapes[36:42], im_size=im_size)
        eye_img_r, eye_rect_r = cef.crop_eye(img=img, gray=gray, eye_points=shapes[42:48], im_size=im_size)

        eye_img_l = cv2.resize(eye_img_l, dsize=im_size)
        eye_img_r = cv2.resize(eye_img_r, dsize=im_size)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)
        
        # janelas de apresentação com olho esquerdo e direito
        cv2.imshow('l', eye_img_l)
        cv2.imshow('r', eye_img_r)

        eye_input_l = eye_img_l.copy().reshape((1, im_size[1], im_size[0], 1)).astype(np.float32) / 255.
        eye_input_r = eye_img_r.copy().reshape((1, im_size[1], im_size[0], 1)).astype(np.float32) / 255.

        pred_l = model.predict(eye_input_l)
        pred_r = model.predict(eye_input_r)
        
        # condições para vizualização
        limiar = 0.01
        state_l = 'aberto' if pred_l > limiar else 'fechado'
        state_r = 'aberto' if pred_r > limiar else 'fechado'
        
        # alarme
        if (state_l == 'fechado' and state_r == 'fechado' and eye_timer == 0):
            eye_timer = time.time()
            mixer.music.pause()
        elif (state_l == 'fechado' and state_r == 'fechado' and eye_timer != 0):
            if (time.time() - eye_timer >= 4):
                cv2.putText(img, "ALARME!!!", (110,230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                mixer.music.play()
        else:
            eye_timer = 0
            mixer.music.pause()
        
        # esboço do retângulo ao redor dos olhos
        if state_l == 'aberto':
            cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(0,128,0), thickness=1)
        else:
            cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(0,0,255), thickness=1)
            
        if state_r == 'aberto':
            cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(0,128,0), thickness=1)
        else:
            cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(0,0,255), thickness=1)
        
        # texto indicando o estado dos olhos
        cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
        
    # janela de apresentação com a face
    cv2.imshow('img',img)
    i += 1
    # tecle 'q' para quebrar o loop
    if(cv2.waitKey(1) == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




