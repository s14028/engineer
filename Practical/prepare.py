from common import *
import numpy as np
import scipy.io as mat

def read_anwsers(y_file):
	y = mat.loadmat(y_file)
	person_coo_tensor = np.array([i[0][0][0] for i in y["frame"][0]])
	count_vector = np.array([i[0] for i in y["count"]])
	
	for matrix in person_coo_tensor:
		matrix[:, :] = matrix[:, ::-1]
	
	return person_coo_tensor, count_vector

def sort_anwsers(anwsers):
	for index, array in enumerate(anwsers):
		np.round(array)
		array = sorted(array, key=lambda element: element[0])
		array = sorted(array, key=lambda element: element[1])

		anwsers[index] = array

def create_dirs():
	directories = directory_names()

	for directory in directories:
		os.mkdir(directory)
	return directories

def copy(data, anwsers, directory, indexes):
	matrix, counts = anwsers
	count = counts.shape[0]
	
	matrix = matrix[indexes]
	counts = counts[indexes]
	
	np.save("{}_anwsers".format(directory), matrix)
	np.savetxt("{}_count.csv".format(directory), counts, delimiter=',')

	for i in np.arange(count)[indexes]:
		img.imwrite("{}/image_{}.jpg".format(directory, i), data[i])

def copy_images(data, anwsers, directories):
	matrix, counts = anwsers
	choice = np.zeros((2, matrix.shape[0]), dtype=np.bool)
	choice[1][np.random.choice(matrix.shape[0], (300,), replace=False)] = True
	choice[0] = ~choice[1]

	for index, indexes in enumerate(choice):
		copy(data, anwsers, directories[index + 1], indexes)

def data_sets():
	images = read_images("mall_dataset/frames")
	matrix, counts = read_anwsers("mall_dataset/mall_gt.mat")

	sort_anwsers(matrix)

	return images, (matrix, counts)

def main():
	data, anwsers = data_sets()
	directories = create_dirs()
	copy_images(data, anwsers, directories)


if __name__ == "__main__":
	main()