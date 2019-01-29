from common import *
import numpy as np
import imageio as img
import matplotlib.pyplot as plt
import os
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

	def _part_copier_builder(self, image):
		def part_copier(z, y, x, t):
			return image[((z - z % self._parts) // self._parts) * self._y_length + y, (z % self._parts) * self._x_length + x, t]

		return part_copier

	def _square_split_image(self, image):
		parts_count = self._parts ** 2
		part_copier = self._part_copier_builder(image)
		array = np.fromfunction(part_copier, shape=(parts_count, self._y_length, self._x_length, 3), dtype=np.uint8)
		return array

	def _prepare_images(self, image_tensor):
		images = np.empty((image_tensor.shape[0], self._parts ** 2, self._y_length, self._x_length, 3), dtype=np.uint8)
		for index, image in enumerate(image_tensor):
			images[index] = self._square_split_image(image)
		images = images.reshape((-1, self._y_length, self._x_length, 3))
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

	def fit(self, image_tensor, person_coo_tensor):
		raise NotImplementedError
    
	def predict(self, image_matrix):
		raise NotImplementedError
        
	def predict_proba(self, image_matrix):
		raise NotImplementedError

	def predict_tensor(self, image_tensor):
		raise NotImplementedError
        
	def predict_proba_tensor(self, image_tensor):
		raise NotImplementedError

	def load(self, path):
		raise NotImplementedError

	def save(self, path):
		raise NotImplementedError