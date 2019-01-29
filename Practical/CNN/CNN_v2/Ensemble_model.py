# In[1]:


get_ipython().run_line_magic('cd', '../../')


# In[2]:


import keras
import scipy.io as mat

from common import *
from augmentation import add_pmap
from augmentation import augmentation_data
from augmentation import augment_data
from testing import test_model
from testing import best_cop_diff
from testing import mse
from testing import mae

from CNN.CNN_v2 import CNN_v2
from CNN.EnsembleCNN import EnsembleCNN


# In[3]:


perspective = mat.loadmat("mall_dataset/perspective_roi.mat")["pMapN"]

perspective /= np.min(perspective)
perspective = np.round(perspective).astype(np.uint8)

train, test = data_sets()
image_tensors = train[0], test[0]
person_coo_tensors = train[1], test[1]
count_matrix = train[2], test[2]

image_train, image_test = image_tensors
person_coo_train, person_coo_test = person_coo_tensors
count_train, count_test = count_matrix
count_train = count_train.astype(np.uint16)
count_test = count_test.astype(np.uint16)

image_train = add_pmap(image_train, perspective)
image_test = add_pmap(image_test, perspective)


# In[4]:


cop = np.linspace(0, 1, 11)[1:-1]


# In[5]:


cnn_v2 = CNN_v2((480, 640, 4), split_into_parts=20)

images = cnn_v2._prepare_images(image_train)
anwsers = cnn_v2._prepare_anwsers(person_coo_train)


# In[6]:


ones_count = (np.sum(anwsers == 1) * 0.25).astype(np.uint32)
zeros_count = (ones_count / 0.25 * 0.75).astype(np.uint32)
validation_length = (zeros_count + ones_count).astype(np.int32)

val_indices = np.concatenate([np.where(anwsers == 0)[0][:zeros_count],
                              np.where(anwsers == 1)[0][:ones_count]])


# In[7]:


anwsers[val_indices[zeros_count:]] = 0


# In[8]:


val_indices = -(images.shape[0] - val_indices)


# In[9]:


generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=20)

augmentation = augmentation_data(image_train, anwsers, 20)
augmented_data = augment_data(generator, augmentation, images, anwsers)
images, anwsers = augmented_data


# In[10]:


anwsers[val_indices[zeros_count:]] = 1

images[-validation_length:], images[val_indices] = images[val_indices], images[-validation_length:]
anwsers[-validation_length:], anwsers[val_indices] = anwsers[val_indices], anwsers[-validation_length:]


# In[11]:


def ensemble_teach(count, x, y, val):
  models = []
  
  for i in range(count):
    print(f"Begin to train {i}-th model.")
    
    model = CNN_v2((480, 640, 4), split_into_parts=20)
    model.def_model()
    
    model.model = keras.utils.multi_gpu_model(model.model, gpus=2, cpu_merge=False)
    model.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    model.model.optimizer.lr.assign(1e-3)
    model.model.fit(x, y, batch_size=20000, epochs=30, validation_data=val)
    model.model.optimizer.lr.assign(1e-4)
    model.model.fit(x, y, batch_size=20000, epochs=20, validation_data=val)
    model.model.optimizer.lr.assign(5e-5)
    model.model.fit(x, y, batch_size=20000, epochs=80, validation_data=val)
    
    model.model.save_weights(f"CNN/CNN_v2/weights/ensemble_model_{i}")
    print(f"Model {i}-th finished training", end="\n\n")
    
    models.append(model)
    
  return models


# In[ ]:


models = ensemble_teach(9,
               images[:-validation_length],
               anwsers[:-validation_length],
               (images[-validation_length:], anwsers[-validation_length:]))


# In[ ]:


model = EnsembleCNN(models, (480, 640, 4), split_into_parts=20)


# In[ ]:


result = test_model(model=model, images=image_test, counts=count_test, cut_of_points=cop)


# In[ ]:


diff = best_cop_diff(result, count_test)

print(f"Model EnsembleCNN behaved:")

print(f"For cut-of-point {diff[0]} had accuracy {diff[1]}:")
print(diff[2])

print(f"With MSE {mse(diff[2])}")
print(f"With MAE {mae(diff[2])}", end="\n\n")

