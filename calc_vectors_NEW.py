import  os
import subprocess
import numpy as np
from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans
from spherecluster import VonMisesFisherMixture
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt

allClusters = []
allClustersID = []
allIngPhrases  = {}	
class ingphrase:
	def __init__(self,ingphraseID,ingphrase,vector,clusterID):
		self.ingphraseID = ingphraseID
		self.ingphrase = ingphrase
		self.vector = vector
		self.clusterID = clusterID

class cluster:
	def __init__(self,clusterID,ingpharsearr):
		self.clusterID = clusterID
		self.ingpharsearr = ingpharsearr

def scrapeIner():
	lines = []
	xarr = []
	yarr = []
	with open ("inertia.txt") as file:
		lines = file.readlines()
	for line in range(1,len(lines),2):
		x = lines[line].split(" ")
		xarr.append(int(x[0]))
		yarr.append(float(x[1]))
	#print(xarr,yarr)
	return (xarr,yarr)


def makePlot(xarray,yarray):
	"""
	
	"""
	#(xarray,yarray) = scrapeIner()
	plt.plot(xarray, yarray)
	plt.show()

def makeIngph(labels,TaggedLines,vectors):
	"""
	Input -
	Output - 
	"""
	for index in range(len(labels)):
		clusterID = labels[index]
		ingPhrase = TaggedLines[index]
		vector = vectors[index]
		ingphraseID = index
		newIngPhrase = ingphrase(ingphraseID,ingPhrase,vector,clusterID)
		#allIngPhrases.append(newIngPhrase)
		allIngPhrases[index] = newIngPhrase


def GroupClusters(labels,nfclusters):
	"""
	Input - Labels Array like [0,2,5,4,2,3,1] ,nfclusters = 6
	Function - Makes clusters with cluster id e.g = 2 having [1,4] as ingphrasearr
	"""
	for number in labels:
		if number not in allClustersID:
			allClustersID.append(number)
			ingpharsearr =[]
			for index in range(len(labels)):
				if number == labels[index]:
					ingpharsearr.append(index)
			newCluster = cluster(number,ingpharsearr)
			allClusters.append(newCluster)
			if len(allClustersID) == nfclusters:
				break 
def makeDict():
	"""
	Returns a dictionary of all Tags with initial count = 0
	Tag names are stored in tags_meaning file(along with theri meaninings)
	"""
	tagDict = {}
	lines = []
	with open("tags_meaning")as file:
		lines = file.readlines()
	for line in lines:
		key = line.split(" ")[0].strip("\n")
		tagDict[key] = 0
	return tagDict


def SphericalkMeansCluster(X,nfclusters):
	# Find K clusters from data matrix X (n_examples x n_features)

	# spherical k-means
	
	skm = SphericalKMeans(nfclusters)
	skm.fit(X)

	#print(skm.cluster_centers_)
	#print("Labels =")
	#print(skm.labels_)
	#print("Inertia = ")
	#print(nfclusters,skm.inertia_)
	#return skm.inertia_
	return skm.labels_


def kMeansCluster(X,nfclusters):
	# Number of clusters
	kmeans = KMeans(nfclusters)
	# Fitting the input data
	kmeans = kmeans.fit(X)
	# Getting the cluster labels
	labels = kmeans.predict(X)
	# Centroid values
	centroids = kmeans.cluster_centers_
	#print("Centroid values")
	#print("sklearn")
	#print(centroids)
	correct = 0
	predictions = []
	for i in range(len(X)):
		predict_me = np.array(X[i].astype(float))
		predict_me = predict_me.reshape(-1, len(predict_me))
		prediction = kmeans.predict(predict_me)
		predictions.append(prediction[0])
	return predictions

def makeVector(taggedLine):
	"""
	Input - A tagged Line like 
	1_CD pound_NN skinless,_NN boneless_JJ chicken_NN breast_NN halves_NNS -_: cut_VB into_IN chunks_NNS
	
	"""
	tagDict = makeDict()
	vector = []
	for key in tagDict.keys():
		countkey = taggedLine.count(key)
		tagDict[key] = countkey
		vector.append(countkey)
	#print(vector)
	return vector



if __name__ == "__main__":
	
	#Tags each sentence by the pretrained java model 
	
	#print("Tagging each Line .............")
	#os.system("java -jar twitie_tag.jar models/gate-EN-twitter.model cleangeniuskitcheningrephrases.txt 2>&1 | tee Taggedcleangeniuskitcheningrephrases.txt")
	# proc = subprocess.Popen("java -jar twitie_tag.jar models/gate-EN-twitter.model AllIngredients.txt", shell=True, stdout=subprocess.PIPE, )
	# #Tagged Lines stored by the java model
	
	# TaggedLines = ((proc.communicate()[0]).decode("utf-8")).split("\n")
	

	# Tagged Lines
	TaggedLines = []
	with open ("Taggedcleangeniuskitcheningrephrases.txt") as file1:
		lines1 = file1.readlines()
		for line1 in lines1:
			line1 = line1.strip("\n")
			TaggedLines.append(line1)

	print("Tagged all Lines and received in python program")
	#print(TaggedLines)


	#Generating corresponding vector to each TaggedLine
	testSetVectors = []
	for taggedline in TaggedLines:
		testSetVectors.append(makeVector(taggedline))
	testSetVectors = np.array(testSetVectors)
	
	#print("Range of Every Column = ",testSetVectors.ptp(0))
	#ptp(0) gives max - min
	testSetVectorsNormed = testSetVectors/testSetVectors.ptp(0)
	
	testSetVectorsNormed[np.isnan(testSetVectorsNormed)] = 0
	testSetVectorsNormed[np.isinf(testSetVectorsNormed)] = 1

	# print("Original = ",np.transpose(testSetVectors))
	# print("Normalized  = ",np.transpose(np.around(testSetVectorsNormed,2)))
	# print("Shape of Normalized Matrix  = ",np.shape(np.array(testSetVectorsNormed)))
	#x = []
	nfclusters = 25
	#inertias = []
	
	labels = SphericalkMeansCluster(testSetVectorsNormed,nfclusters)
	#inertias.append(labels)

	#makePlot(x,inertias)
	makeIngph(labels,TaggedLines,testSetVectorsNormed)
	

	#See All Ingredient phrases side by side
	lines3 = []
	with open("cleangeniuskitcheningrephrases.txt") as file3:
		lines3 = file3.readlines()
		for i in range(len(lines3)):
			print(lines3[i].strip("\n"),allIngPhrases[i].ingphrase)
	
	GroupClusters(labels,nfclusters)
	nftrainingphr = 0
	all_ingredient_phrase = []
	all_ingredient_phrase_train = []
	ingredient_set = set({})
	ingredient_set_train = set({})
	with open("gktrain.txt") as file4:
		train_lines = file4.readlines()
	print(" Ingredient Phrases : ")
	
	for line1 in train_lines:
		line1 = line1.strip("\n")
		all_ingredient_phrase_train.append(line1)
		ingredient_set_train.add(line1)

	for cluster in allClusters:
		list1 = random.choices(cluster.ingpharsearr, k = int( 1/2500 * len(cluster.ingpharsearr)))
		print("----------------------------------------------------------------------------")
		print("Number of Ingredient phrases chosen from each cluster is =",len(list1))
		nftrainingphr = nftrainingphr + len(list1)
		for ingphID in list1:
			li = lines3[ingphID].strip("\n")
			while ((li in all_ingredient_phrase) or (li in all_ingredient_phrase_train)):
				print("DUPLICATE")
				print(li)
				ing4  = random.choices(cluster.ingpharsearr,k = 1)
				ingphID = ing4[0]
				li = lines3[ingphID].strip("\n")
				print(li)
			print(li) 
			print(ingphID)
			all_ingredient_phrase.append(li)
			ingredient_set.add(li)

	print(len(all_ingredient_phrase))
	print(nftrainingphr)
	print(len(ingredient_set))
	with open("gktest1.txt","w") as file2:
		for phrase in all_ingredient_phrase:
			print(phrase)
			file2.write(phrase)
			file2.write("\n")


