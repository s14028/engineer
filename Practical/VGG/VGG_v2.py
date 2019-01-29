from base_model import *
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

class VGG_v2(ImagePersonCounter):

	def __init__(self, shape, split_into_parts=8, cut_off_point=0.4):
		super().__init__((224, 224, shape[2]), split_into_parts, cut_off_point)
		self._shape = shape
		self._part_length = self._x_length
    
	def _prepare_images(self, image_tensor):
		images_for_y = np.ceil((self._shape[0] - 224) / 224).astype(np.uint8) + 1
		images_for_x = np.ceil((self._shape[1] - 224) / 224).astype(np.uint8) + 1

		images = np.empty((image_tensor.shape[0] * images_for_x * images_for_y, 224, 224, self._z_length), dtype=np.uint8)

		for i in range(images.shape[0]):
			index = i // (images_for_x * images_for_y)
			part_index = i % (images_for_x * images_for_y)

			y_part = part_index // images_for_x
			x_part = part_index % images_for_x

			y_index = y_part * 224
			x_index = x_part * 224

			if y_part == images_for_y - 1:
				y_index = self._shape[0] - 224

			if x_part == images_for_x - 1:
				x_index = self._shape[1] - 224

			images[i] = image_tensor[index, y_index:y_index + 224, x_index:x_index + 224]

		return images

	def _double_section(self, image_count, length):
		return (length - 224, (image_count - 1) * 224)

	def _indices(self, value, length, double_section):
		value_left, value_right = -1, -1
		section_left, section_right = -1, 0

		if value < double_section[1]:
			section_left = value // 224
			section_right = 1
            
			value_left = value % 224
			value_left = value_left // self._part_length
            
		if value >= double_section[0]:
			section_right += value // 224
            
			value_right = 224 - (length - value)
			value_right = value_right // self._part_length
            
		else:
			section_right = -1

		return (value_left, value_right), (section_left, section_right)


	def _prepare_anwsers(self, person_coo_tensor):
		images_for_x = np.ceil((self._shape[1] - 224) / 224).astype(np.uint8) + 1
		images_for_y = np.ceil((self._shape[0] - 224) / 224).astype(np.uint8) + 1

		anwsers = np.zeros((len(person_coo_tensor) * images_for_x * images_for_y, self._parts ** 2), dtype=np.uint8)

		double_section_x = self._double_section(images_for_x, self._shape[1])
		double_section_y = self._double_section(images_for_y, self._shape[0])

		for i, person_coo_matrix in enumerate(person_coo_tensor):
			index = i * images_for_x * images_for_y
			for y, x in person_coo_matrix:
				x = int(round(x))
				y = int(round(y))

				(x_left, x_right), (xs_left, xs_right) = self._indices(x, self._shape[1], double_section_x)
				(y_left, y_right), (ys_left, ys_right) = self._indices(y, self._shape[0], double_section_y)

				if xs_left != -1:
					if ys_left != -1:
						anwsers[index + ys_left * images_for_x + xs_left, y_left * self._parts + x_left] = 1
					if ys_right != -1:
						anwsers[index + ys_right * images_for_x + xs_left, y_right * self._parts + x_left] = 1
				if xs_right != -1:
					if ys_left != -1:
						anwsers[index + ys_left * images_for_x + xs_right, y_left * self._parts + x_right] = 1
					if ys_right != -1:
						anwsers[index + ys_right * images_for_x + xs_right, y_right * self._parts + x_right] = 1

		return anwsers
        
	def def_model(self):
		model = keras.Sequential()
		sub_model = VGG19(include_top=False, input_shape=(224, 224, self._z_length))

		for layer in sub_model.layers:
			layer.trainable = False
        
		model.add(sub_model)
		model.add(keras.layers.Flatten())
        
		model.add(keras.layers.Dense(256, activation="relu"))
		model.add(keras.layers.Dropout(0.2))
		model.add(keras.layers.Dense(128, activation="relu"))
		model.add(keras.layers.Dropout(0.1))
		model.add(keras.layers.Dense(64, activation="sigmoid"))

		model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

		self.model = model

		return model
        
	def axis_length(self, double_section, image_count):
		section_diff = (double_section[1] - double_section[0])
		pixel_loss = section_diff % (224 // self._parts)
		double_section_parts = (section_diff // (224 // self._parts))

		length = image_count * self._parts - double_section_parts - (1 if pixel_loss < 5 else 0)

		return length, double_section_parts + (1 if pixel_loss < 5 else 0)
    
	def predict(self, image_tensor):
		return np.sum(self.predict_proba_tensor(image_tensor) > self._cut_off_point, axis=1)
    
	def predict_proba(self, image_tensor):
		images = self._prepare_images(image_tensor)
        
		predictions = self.model.predict(images)
		predictions = predictions.reshape((-1, self._parts, self._parts))

		images_for_y = np.ceil((self._shape[0] - 224) / 224).astype(np.uint8) + 1
		images_for_x = np.ceil((self._shape[1] - 224) / 224).astype(np.uint8) + 1

		double_section_y = self._double_section(images_for_y, self._shape[0])
		double_section_x = self._double_section(images_for_x, self._shape[1])

		y_length, double_section_y = self.axis_length(double_section_y, images_for_y)
		x_length, double_section_x = self.axis_length(double_section_x, images_for_x)
        
		shape = (image_tensor.shape[0], y_length, x_length)
        
		consistent_predictions = np.empty(shape, dtype=np.float64)

		for index, image_pred in enumerate(predictions):
			new_index = index // (images_for_x * images_for_y)
			index %= images_for_x * images_for_y
            
			y_part = index // images_for_x
			x_part = index % images_for_x

			left_yb = y_part * self._parts
			left_ye = left_yb + self._parts
			left_xb = x_part * self._parts
			left_xe = left_xb + self._parts

			right_y = 0
			right_x = 0

			if y_part == images_for_y - 1:
				left_ye = y_length
				right_y = double_section_y

			if x_part == images_for_x - 1:
				left_xe = x_length
				right_x = double_section_x

			consistent_predictions[new_index, left_yb:left_ye, left_xb:left_xe] = image_pred[right_y:, right_x:]

		consistent_predictions = consistent_predictions.reshape((-1, y_length * x_length))

		return consistent_predictions

	def load(self, path):
		dummy_image = np.empty((1, self._shape[0], self._shape[1], self._z_length), dtype=np.uint8)
		dummy_coordinates = np.array([[[0, 0]]])
		self.fit(dummy_image, dummy_coordinates)
		self.model.load_weights(path)

	def save(self, path):
		self.model.save_weights(path)