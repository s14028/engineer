from common import *
import scipy.io as mat
import keras.preprocessing

def add_pmap(images, perspective_map):
  new_images = np.empty(np.concatenate([images.shape[:-1], [4]]), dtype=np.uint8)
  new_images[:, :, :, :-1] = images
  new_images[:, :, :, -1] = perspective_map
  
  return new_images

def add_padding(images, padding_shape, fill_value=0):
  shape = np.array(images.shape)
  padding_shape = np.array(padding_shape, dtype=np.int16)
  
  shape[1:-1] += padding_shape * 2
  
  padded_images = np.full(shape, fill_value=fill_value)
  
  for index, image in enumerate(images):
    padded_images[index,
                  padding_shape[0]:-padding_shape[0],
                  padding_shape[1]:-padding_shape[1]] = image
    
  return padded_images

# Returns offset and side length of images which will be augmented
def image_new_coo(shape):
  shape = np.array(shape)
  shape = shape / 2
  length = np.sqrt(np.dot(shape, shape))
  return (length - shape).astype(np.uint32), int(length * 2)

# Transforms index of subimage in flat array into indices of original tensor (image)
def transform_index(index, parts):
  image_index = index // parts ** 2
  y = (index % parts ** 2) // parts
  x = (index % parts ** 2) % parts
  
  return image_index, y, x

def tensor_indices(indices, parts):
  new_indices = np.empty((indices.shape[0], 3), dtype=np.uint32)
  for i, index in enumerate(indices):
    new_indices[i] = transform_index(index, parts)
  
  return new_indices

# Transofrm indices of original tensor into specific coordinates
def to_coo(indices, shape):
  indices[:, 1:] *= shape
  return indices

# Copies image tensor into flatten array of images parts which can be augmentable
def copy(images, coo, length):
  new_images = np.empty((coo.shape[0], length, length, images.shape[3]), dtype=np.uint8)

  for index, image in enumerate(coo):
    new_images[index] = images[image[0],
                               image[1]:image[1] + length,
                               image[2]:image[2] + length]
  
  return new_images

# Recieves tensor of images in original shape without splitting and anwsers for parts after image split
def augmentation_data(images, anwsers, parts):
  shape = np.array([images.shape[1] // parts, images.shape[2] // parts], dtype=np.uint16)
  offset, side_length = image_new_coo(shape)
  images = add_padding(images, offset)
  
  augmentable_images = np.where(anwsers == 1)[0]

  augmentable_images = tensor_indices(augmentable_images, parts)

  augmentable_coo = to_coo(augmentable_images, shape)
  augmentable_coo[:, 1:] = augmentable_coo[:, 1:]
  
  return copy(images, augmentable_coo, side_length)

def generate_image(generator, image):
  for i in generator.flow(image, batch_size=1):
    return i
  
def generate_data(generator, images, new_length):
  generated_images = np.empty(np.concatenate([[new_length], images.shape[1:]]), dtype=np.uint8)
  
  image_index = 0
  
  for index in range(new_length):
    if image_index == images.shape[0]:
      image_index = 0
    
    generated_images[index] = generate_image(generator, images[image_index:image_index + 1])
    
    image_index += 1
  
  return generated_images

def augmented_length(fill_value, anwsers, ratio):
  augmentable = anwsers == fill_value
  augmentable_ratio = np.sum(augmentable) / augmentable.shape[0]

  if ratio <= augmentable_ratio or augmentable_ratio == 0 or ratio == 1:
    return -1
  
  new_length = np.round(np.sum(~augmentable) / (1 - ratio)).astype(np.uint32)
  
  return new_length
  

def augment_data(generator, augmentation, images, anwsers, ratio=0.5):
  fill_value = 1
  new_length = augmented_length(fill_value, anwsers, ratio)

  if new_length == -1:
    return np.empty((0,)), np.empty((0,))
  
  generated_images = np.empty(np.concatenate([[new_length], images.shape[1:]]), dtype=np.uint8)
  generated_anwsers = np.empty((new_length,), dtype=np.uint8)
  
  images_to_produce = generated_images.shape[0] - images.shape[0]
    
  augmentation_shape = np.array([augmentation.shape[1], augmentation.shape[2]])
  shape = np.array([images.shape[1], images.shape[2]])
  offset = ((augmentation_shape - shape) // 2).astype(np.int16)
  
  generated_data = generate_data(generator, augmentation, images_to_produce)
  generated_data = generated_data[:, offset[0]:-offset[0], offset[1]:-offset[1]]
  
  generated_images[:images_to_produce] = generated_data
  generated_images[images_to_produce:] = images
  
  generated_anwsers[images_to_produce:] = anwsers
  generated_anwsers[:images_to_produce] = fill_value
  
  return generated_images, generated_anwsers