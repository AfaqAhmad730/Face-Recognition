#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pickle
import face_recognition


# In[12]:


# Function to load the model from a file
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)


# In[13]:


# Step 6: Model inference
def recognize_face(clf, face_encoding):
    return clf.predict([face_encoding])[0]


# In[15]:


# Use the trained classifier for new predictions
new_image = face_recognition.load_image_file('test/elon.jpg')
new_face_encoding = face_recognition.face_encodings(new_image)[0]
clf = load_model('face_recognition_model.pkl')
result = recognize_face(clf, new_face_encoding)
print(result)


# In[ ]:





# In[ ]:




