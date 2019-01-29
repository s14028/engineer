from base_model import *

class EnsembleCNN(ImagePersonCounter):

	def __init__(self, models, shape, split_into_parts=20, cut_off_point=0.4):
		super().__init__(shape, split_into_parts, cut_off_point)
		self.models = models
        
	def def_model(self):
		raise NotImplementedError
    
	def predict(self, image_tensor):
		return np.sum(self.predict_proba_tensor(image_tensor) > self._cut_off_point, axis=1)
    
	def predict_proba(self, image_tensor):
		images = self._prepare_images(image_tensor)
		prediction_matrix = np.empty((len(self.models), images.shape[0]), np.float64)
        
		for index, model in enumerate(self.models):
			prediction_matrix[index] = model.model.predict(images, batch_size=20000).reshape((-1,))

		predictions = np.mean(prediction_matrix, axis=0)
		predictions = predictions.reshape((image_tensor.shape[0], self._parts ** 2))
        
		return predictions