import imageio as img
import numpy as np
import os

def directory_names():
	with open("directory_names.txt", "r") as file:
		directories = file.readlines()
	directories = [i[:-1] for i in directories]
	directories[1:] = [directories[0] + "/" + i for i in directories[1:]]

	return directories

# Recursively iterates through directory and yields files which satisfyies pattern.
def yield_files(directory, pattern):
	for dir_name, dir_list, file_list in os.walk(directory):
		for file in file_list:
			file = str(file)
			if pattern(file):
				yield(dir_name + "/" + file)

def count_files_dir(directory, pattern):
	# patter is lambda like lambda name: name.endswith("")
	counter = 0
	for i in yield_files(directory, pattern):
		counter += 1
	
	return counter

def read_images(directory):
	pattern = lambda name: name.endswith(".jpg")
	number_of_elements = count_files_dir(directory, pattern)
	images = np.empty((number_of_elements, 480, 640, 3), dtype=np.uint8)
	
	for index, file in enumerate(yield_files(directory, pattern)):
		images[index] = img.imread(file)
	
	return images

def load(directory):
	matrix = np.load("{}_anwsers.npy".format(directory), encoding="latin1")
	counts = np.loadtxt("{}_count.csv".format(directory))
	data = read_images(directory)
	
	return (data, matrix, counts)

def data_sets():
	directories = directory_names()

	train = load(directories[1])
	test = load(directories[2])

	return train, test