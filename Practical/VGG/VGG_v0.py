from base_model import *
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

class VGG_v0(ImagePersonCounter):

	def __init__(self, shape, split_into_parts=20, cut_off_point=0.4):
		super().__init__(shape, split_into_parts, cut_off_point)

# In this version i've been using simple splitting of image and placing it in specific place on 224 x 224 image.
# Like so:
#  [ splitted_image * * ]
#  [ * * * ]
#  [ * * * ]
	def _prepare_images(self, image_tensor):
		images = np.empty((image_tensor.shape[0] * self._parts ** 2, 224, 224, self._z_length), dtype=np.uint8)
		y_length = (224 - self._y_length) / (self._parts - 1)
		x_length = (224 - self._x_length) / (self._parts - 1)

		for i in range(image_tensor.shape[0]):
			for y in range(0, self._y_length * self._parts, self._y_length):
				for x in range(0, self._x_length * self._parts, self._x_length):
					index_z = i * self._parts ** 2 + (y // self._y_length) * self._parts + (x // self._x_length)
					index_x = round((x // self._x_length) * x_length)
					index_y = round((y // self._y_length) * y_length)
    
					images[index_z, index_y:index_y + self._y_length, index_x:index_x + self._x_length] = image_tensor[i, y:y + self._y_length, x:x + self._x_length]

		return images
    
	def def_model(self):
		model = keras.Sequential()
		sub_model = VGG19(include_top=False, input_shape=(224, 224, self._z_length))

		for layer in sub_model.layers:
			layer.trainable = False
        
		model.add(sub_model)
		model.add(keras.layers.Flatten())
        
		model.add(keras.layers.Dense(256, activation="relu"))
		model.add(keras.layers.Dense(128, activation="relu"))
		model.add(keras.layers.Dense(128, activation="relu"))
		model.add(keras.layers.Dense(64, activation="relu"))
		model.add(keras.layers.Dense(32, activation="relu"))
		model.add(keras.layers.Dense(16, activation="relu"))
		model.add(keras.layers.Dense(2, activation="relu"))
		model.add(keras.layers.Dense(1, activation="sigmoid"))

		model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

		self.model = model
        
		return model
    
	def predict(self, image_matrix):
		return np.sum(self.predict_proba(image_matrix) > self._cut_off_point)

	def predict_proba(self, image_matrix):
		images = self._square_split_image(image_matrix)

		predictions = self.model.predict(images)

		return predictions
    
	def predict_tensor(self, image_tensor):
		return np.sum(self.predict_proba_tensor(image_tensor) > self._cut_off_point, axis=1)
    
	def predict_proba_tensor(self, image_tensor):
		images = self._prepare_images(image_tensor)

		predictions = self.model.predict(images)
		predictions = predictions.reshape((image_tensor.shape[0], self._parts ** 2))
        
		return predictions

	def load(self, path):
		dummy_image = np.empty((1, self._y_length * self._parts, self._x_length * self._parts, self._z_length), dtype=np.uint8)
		dummy_coordinates = np.array([[[0, 0]]])
		self.fit(dummy_image, dummy_coordinates)
		self.model.load_weights(path)

	def save(self, path):
		self.model.save_weights(path)