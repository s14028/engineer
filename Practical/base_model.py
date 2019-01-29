import numpy as np
import matplotlib.pyplot as plt
import keras
import keras.layers

class ImagePersonCounter:
    
	_parts: int
	_y_length: int
	_x_length: int

	def __init__(self, shape, split_into_parts=20, cut_off_point=0.4):
		self._parts = split_into_parts
		self._cut_off_point = cut_off_point
		
		self._y_length = shape[0] // self._parts
		self._x_length = shape[1] // self._parts
		self._z_length = shape[2]

	def _prepare_images(self, image_tensor):
		images = np.empty((image_tensor.shape[0] * self._parts ** 2, self._y_length, self._x_length, self._z_length), dtype=np.uint8)
        
		for i in range(image_tensor.shape[0]):
			for y in range(0, self._y_length * self._parts, self._y_length):
				for x in range(0, self._x_length * self._parts, self._x_length):
					index = i * self._parts ** 2 + (y // self._y_length) * self._parts + (x // self._x_length)
    
					images[index] = image_tensor[i, y:y + self._y_length, x:x + self._x_length]
        
		return images

	def _prepare_anwser_vector(self, person_coo_matrix):
		anwser_vector = np.zeros((self._parts ** 2,))
		
		for y, x in person_coo_matrix:
			y_parts = y // self._y_length
			x_parts = x // self._x_length
			index = int(x_parts + y_parts * self._parts)
			anwser_vector[index] = 1
		
		return anwser_vector

	def _prepare_anwsers(self, person_coo_tensor):
		anwsers = np.empty((person_coo_tensor.shape[0], self._parts ** 2))
		
		for index, person_coo_matrix in enumerate(person_coo_tensor):
			anwsers[index] = self._prepare_anwser_vector(person_coo_matrix)
			
		anwsers = anwsers.reshape((-1,))

		return anwsers
    
	def def_model(self):
		raise NotImplementedError

	def fit(self, x, y, batch_size=1, epochs=1, callbacks=[], validation_split=0.0):
		x = self._prepare_images(x)
		y = self._prepare_anwsers(y)
        
		self.model.fit(x, y,
					batch_size=batch_size,
					epochs=epochs,
					callbacks=callbacks,
					validation_split=validation_split)

	def predict(self, image_tensor):
		raise NotImplementedError
        
	def predict_proba(self, image_tensor):
		raise NotImplementedError

	def load(self, path):
		raise NotImplementedError

	def save(self, path):
		raise NotImplementedError