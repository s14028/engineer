from base_model import *

class SDM(ImagePersonCounter):

	def __init__(self, shape, split_into_parts=20, cut_off_point=0.4):
		super().__init__(shape, split_into_parts, cut_off_point)
    
	def def_model(self):
		model = keras.Sequential()

		model.add(keras.layers.BatchNormalization(input_shape=(self._y_length * self._x_length * self._z_length,)))
		model.add(keras.layers.Dense(128, activation="relu"))

		model.add(keras.layers.BatchNormalization())
		model.add(keras.layers.Dense(1, activation="sigmoid"))

		model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

		self.model = model

		return model
    
	def _prepare_images(self, image_tensor):
		images = super()._prepare_images(image_tensor)
		images = images.reshape((-1, self._y_length * self._x_length * self._z_length))
        
		return images

	def predict(self, image_tensor):
		return np.sum(self.predict_proba_tensor(image_tensor) > self._cut_off_point, axis=1)
    
	def predict_proba(self, image_tensor):
		images = self._prepare_images(image_tensor)

		images = images.reshape((-1, self._y_length * self._x_length * self._z_length))
		predictions = self.model.predict(images, batch_size=615000)

		predictions = predictions.reshape((image_tensor.shape[0], self._parts ** 2))
        
		return predictions

	def load(self, path):
		dummy_image = np.empty((1, self._y_length * self._parts, self._x_length * self._parts, self._z_length), dtype=np.uint8)
		dummy_coordinates = np.array([[[0, 0]]])
		self.fit(dummy_image, dummy_coordinates)
		self.model.load_weights(path)

	def save(self, path):
		self.model.save_weights(path)