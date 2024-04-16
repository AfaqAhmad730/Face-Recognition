#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pickle
import numpy as np


# In[6]:


# Function to load the model from a file
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)


# In[7]:


# Step 6: Model inference
def recognize_face(clf, face_encoding):
    return clf.predict([face_encoding])[0]


# In[12]:


# Use the trained classifier for new predictions
#new_image = face_recognition.load_image_file('test/adele.jpg')
#new_face_encoding = face_recognition.face_encodings(new_image)[0]

new_face_encoding = np.load('test/elon.npy')
#new_face_encoding = np.load('test/amir.npy')
#new_face_encoding = np.load('test/adele.npy')
#new_face_encoding = np.load('test/alia.npy')
#new_face_encoding = np.load('test/obama.npy')
#new_face_encoding = np.load('test/enrique.npy')

clf = load_model('face_recognition_model.pkl')
result = recognize_face(clf, new_face_encoding)
print(result)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




