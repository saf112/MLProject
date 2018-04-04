import numpy as np
import keras
from keras.models import Sequential
import scipy.io
from keras.utils import np_utils
from keras.layers import Dense

#split the data into training and testing data
#inputs: data to split, corresponding labels, fraction to split the data
#output: train data/labels & test data/labels
def split_data(data,labels,frac):
	n=data.shape[0]
	indices=np.array((range(n)))

	#pick frac of indicies for test data
	num_choose=np.int(frac*n)
	rand_ind=np.random.choice(indices,num_choose,replace=False)

	#seperate test data
	test_data=data[rand_ind]
	test_labels=labels[rand_ind]

	#delete test data from training data
	train_data=np.delete(data,rand_ind,axis=0)
	train_labels=np.delete(labels,rand_ind)

	return train_data,train_labels,test_data,test_labels

#train the model 
#input: training data & labels
#output: trained model
def train_model(train,labels):
	num_samples,num_features=train.shape

	model = Sequential()

	#substract 1 from labels in order for the labels to be 0,1,2 instead of 1,2,3
	labels=labels-1
	#transform to one hot labels
	one_hot_labels=np_utils.to_categorical(labels, num_classes=3)

	#add layers
	model.add(Dense(units=64, activation='relu', input_dim=num_features))
	model.add(Dense(units=3, activation='softmax'))


	model.compile(loss='categorical_crossentropy',
				  optimizer='sgd',
				  metrics=['accuracy'])


	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

	#train model
	model.fit(train, one_hot_labels, epochs=1000, batch_size=32)

	return model

#TEST MODEL HELPER FUNCTIONS
#converts one hot labels back to a vector of stages
#input: one hot label matrix
#output: stage labels
def categorical_to_vector(mat):
	r,c=mat.shape
	vec=np.zeros((r,))

	#assign label based on what column the 1 lies
	for i in range(r):
		for j in range(c):
			if mat[i][j]==1:
				vec[i]=j
	#add 1 in order for it it to be a stage label 1,2,3 instead of 0,1,2
	vec=vec+1
	return vec

#calculate accuracy 
def calc_acc(pred,true):
	n=pred.shape[0]
	correct=0
	for i in range(n):
		if pred[i]==true[i]:
			correct+=1
	acc=(correct/float(n))*100
	return acc

#test model
#input: trained model, testing data and labels, batch size
#output: accuracy
def test_model(model,test,labels,batch_size):
	#predict one hot labels from data
	seq_predict=model.predict(test,batch_size=batch_size)
	#convert one hot label predictions to stage labels
	labels_pred=categorical_to_vector(seq_predict)
	#calculate accuracy
	acc=calc_acc(labels_pred, labels)
	return acc

if __name__=="__main__":
	#params for us to play with
	frac=0.1
	batch_size=128

	##GET DATA -- EITHER SAMPLE OR ALL DATA 
	#sample data
	indices=scipy.io.loadmat('sample_indices.mat')['sample_indices']
	labels=scipy.io.loadmat('sample_labels.mat')['sample_labels']
	patient=scipy.io.loadmat('sample_pat.mat')['sample_pat']
	patient=patient.T
	#all data
	# labels=scipy.io.loadmat('labels.mat')['labels']
	# patient=scipy.io.loadmat('features.mat')['features']
	# patient=patient.T

	#split train vs test data
	train_data,train_labels,test_data,test_labels=split_data(patient,labels,frac)

	#train the model
	model=train_model(train_data,train_labels)
	print('trained model')

	#test training data
	train_acc=test_model(model,train_data,train_labels,batch_size)
	print('training acc: ', train_acc)

	#test test data
	test_acc=test_model(model,test_data,test_labels,batch_size)
	print('test accuracy: ', test_acc)

