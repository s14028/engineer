import numpy as np

def accuracy(predictions, anwsers):
	return np.clip(1 - (np.abs(predictions - anwsers) / anwsers), 0, 1)

def test_model(model, images, counts, cut_of_points):
	probabilities = model.predict_proba(images)
	counts = counts.astype(np.int16)
	results = {}

	for cof in cut_of_points:
		predictions = np.sum((probabilities > cof), axis=1)
		predictions = predictions.astype(np.int16)
		acc = np.mean(accuracy(predictions, counts)) * 100
        
		results[cof] = acc, predictions
    
	return results

def best_cop_diff(results, counts):
	best_cop = [-1, -1, None]
	for key, value in results.items():
		accuracy = value[0]
		if accuracy > best_cop[1]:
			best_cop[0] = key
			best_cop[1] = accuracy
			best_cop[2] = value[1]

	best_cop[2] = best_cop[2] - counts
	return best_cop

def mse(diff):
	return np.mean(diff ** 2)

def mae(diff):
	return np.mean(np.abs(diff))