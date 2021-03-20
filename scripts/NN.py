import numpy as np
import random 
import math
from Bio import SeqIO


##### ----------------------------- FUNCTIONS FOR DNA PROCESSING --------------------------------------------------  #####


def oneHotDNA(seq, flatten = True):
	'''One hot encoding for DNA sequences as a four bit vector. 

	Parameters: 
		seq (str): DNA sequence input
		flatten (bool): whether to flatten to 1D structure or not. 
	'''

	mapping = dict(zip("ACGT", range(4)))   #{'A': 0, 'C': 1, 'G': 2, 'T': 3}
	seq2 = [mapping[i] for i in seq] #number in dictionary for all NT listed above
	mat = np.eye(4)[seq2]
	mat = mat.astype(int)
	if flatten == False:
		return mat #transfer into 4 bit matrix. 
	else:
		return mat.flatten()


def process_positives(NL_delim_file, method="simple", window = (3,8,4)):
	'''Processing of input positive yeast sequence

	| This function will read in provided newline delimited file (with no header) and transform all DNA sequences into a 1D hot-encoded representation. 
	| In method "simple" this will take the entire length provided. 
	| In method "sliding" this will generate a sliding window of specified length over every entry prior to 1D transformation. 

	Parameters: 
		NL_delim_file (str): new line delimited file to read in of sequnences. 
		method (str): default ("simple"). "simple" will process all inputs lines as it. "sliding" will enact a sliding window
		window (np array): default ((3,8,4)). Enacted if method == "sliding". Follows the format (No total pieces, final length, overlap length)
	'''
	input_nodes = []
	if method == "simple":
		for line in open(NL_delim_file, "r"):
			seq = line.strip()
			input_vals = oneHotDNA(seq, flatten=True)
			input_nodes.append(input_vals)
		return(input_nodes)
	
	elif method == "sliding":
		for line in open(NL_delim_file, "r"):
			seq = line.strip()
			for i in range(0,window[0]):
				start = i*window[2]
				stop = start+window[1]
				seq_sub = seq[start:stop]
				input_vals = oneHotDNA(seq_sub, flatten=True)
				input_nodes.append(input_vals)
		return(input_nodes)
	else:
		print("Incorrect method selection"); exit()



def process_negatives(fastafile, downsample = None, len_k = 17, record=True, seed = 10):
	'''Processing of input negative yeast sequence

	| This function will read in provided fasta file, optionally downsampmle entries, select a random start site within the selected sequence, and extend to len_k. 
	| The final DNA subsequence will be transformed into a 1D hot-encoded representation. 	

	Parameters: 
		fastafile (str): input fasta file to parse
		downsample (int): default None. How many input DNA sequences to randomly select
		len_k (int): length of random subsequence to pull from DNA seq
		record (bool): default True. Whether to return which DNA sequences were used as input and the random start sites. 
		seed (int): default 10. random seed for reproducibility
	'''

	input_nodes = []
	random.seed(seed)
	tmp = SeqIO.parse(open(fastafile),'fasta')
	fasta_seqs = SeqIO.to_dict(tmp)
	if (downsample == None): keys = list(fasta_seqs)
	else: keys = random.sample(list(fasta_seqs), downsample)

	starts = []
	values = [fasta_seqs[k] for k in keys]
	#fasta_seqs_sub =  dict(zip(keys, values))
	for val in values:
		DNAseq = str(val.seq)
		start_pos = random.randint(0,len(DNAseq) - len_k)
		starts.append(start_pos)
		kmer = DNAseq[start_pos:start_pos+len_k]
		input_vals = oneHotDNA(kmer, flatten=True)
		input_nodes.append(input_vals)

	if (record ==True): return(keys, starts, input_nodes)
	else: return(input_nodes)


def check_negatives(positives, negatives):
	'''Check if any negative sequences appear in positives. 

	| Function returns "NEED TO REMOVE" if there is an error (this has yet to happen)	or "Good to go" if all values are unique. 

	Parameters: 
		positives (np array): np array of either int for float
		negatives (np array): np array of either int for float
	'''

	for i in range(0, len(negatives)-1):
		for j in range(0, len(positives)-1):
			if (np.array_equal(negatives[i], positives[j])):
				print("NEED TO REMOVE")
	print("Good to go")




##### -----------------------------	END --------------------------------------------------  #####










##### -----------------------------	END --------------------------------------------------  #####





class NeuralNetwork(object):
	'''Neural Network class build for simple predictions using one hidden layer. 

	Parameters: 
		setup_nodes (np array): default (8,3,8). The number of nodes in each of the three layers (input, hidden, output). The input layer much match the dimensions of the expected input data used in fit(). The program is not adpated for multiple hidden layers yet. 
		activation (bool): default "sigmoid". Activation funtion for all layers. options are "sigmoid", "relu", "tanh"
		seed (int): default 1. Seed for random weight initialization to ensure reproducible values. 
	Returns:
		initialized network (of class Neural Network)
	'''

	def __init__(self, setup_nodes = (8,3,8), activation = "sigmoid", seed=1):
		self.input_size = setup_nodes[0]  # number of input nodes
		self.hidden_size = setup_nodes[1]  # number of nodes in the hidden layer
		self.output_size = setup_nodes[2]  # number of nodes to return
		self.activation_method = activation

		#Initialize all begining weights and biases to small (+/-) values. 
		np.random.seed(seed)
		self.W1 = np.random.randn(self.input_size, self.hidden_size) *0.5 #randn so theyre negative rather than rand
		self.W2 = np.random.randn(self.hidden_size, self.output_size) *0.5 #added 0.5 to match Erins doc. 
		self.bias1 = np.random.randn(1, self.hidden_size) 
		self.bias2 = np.random.randn(1, self.output_size) 


	def activation_fnct(self, x):
		'''Activation functions. 

		| Function takes layer x, a matrix, and calculates the activation function. It will call "self.activation_method" to select which calculation. Current options are sigmoid, relu, tanh

		Parameters: 
			x (np array)
		'''

		if self.activation_method=="sigmoid":
			return 1.0 / (1.0 + np.exp(-x))
		elif self.activation_method == "relu":
			#return x * (x > 0)
			return np.maximum(0,x)
		elif self.activation_method == "tanh":
			#val = 2*x
			#(2.0 / (1.0 + np.exp(-val))) - 1
			return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
		else:
			print("not adapted for other activation functions"); exit()


	def derivative_fnct(self, x):
		'''Derivative of the activation functions. 

		| The function calculates the slope of the neuron’s output value. Similarly calls self.activation_method

		Parameters: 
			x (np array)
		'''

		if self.activation_method=="sigmoid":
			#return x * (1.0 - x)
			return np.exp(-x)/((1+np.exp(-x))**2) #amanda I think you could change this to np.exp2 in the future. 
		
		elif self.activation_method == "relu":
			x[x<=0] = 0 # the slope for negative values is zero
			x[x>0] = 1 #the slope for positve values is 1 
			return x
		elif self.activation_method == "tanh":
			t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
			return (1-t**2)
		else:
			print("not adapted for other activation functions"); exit()


	def feedforward(self, input_data, report=False):
		'''Forward pass through the network. 

		| This function calculates the hidden_z layer (L2), hidden_activation (A2), output_z (L3), and output_activation (yhat)

		Parameters: 
			input_data (np array): input data to flow through network. 
			report (bool): default False. Used for testing, will output all values. 
		Returns:
			current predictions for output layer. 
		'''

		#math reference: https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
		
		self.L2 = np.dot(input_data, self.W1) + self.bias1  #input layer * first weight matrix + bias of that layer
		self.A2 = self.activation_fnct(self.L2) #activation of first layer weight
		self.L3 = np.dot(self.A2, self.W2) + self.bias2  #from the activation values of hidden, calculate weights to output layer. 

		yhat = self.activation_fnct(self.L3) #given current structure, what is our predictions.

		if report == True:
			return self.L2, self.A2, self.L3, yhat
		else:
			return yhat




	def backprop(self, input_data, expected_output, lr=0.1):
		'''Backward propogation of information from output to hidden layers. 

		| This function calculates the current error between the expected ouput (y) and the models current predictions for it (yhat). 
		| Next, the change in layer 3 based on that error is calculated (delta out). 
		| Followed by our new layer 3 weights. 

		| Given our new output deltas, we can calculate the new hidden deltas and finally the current hidden layer weights. 

		Parameters: 
			input_data (np array): input data to flow through network. 
			expected_output (np array): expected final output (classes in NN or identical to input_data in autoencoder)
			lr (float): default 0.1
		
		:returns: 
			- hidden_weight (np array): Layer2 weights
			- output_weight (np array): Layer3 weights
			- delta_hidden (np array) : Layer2 errors
			- delta_out (np array): Layer3 errors
		'''
		
		#math reference: https://sudeepraja.github.io/Neural/
		#erin reference: 3.10.2021.Algorithms.ipynb

		error = expected_output - self.yhat
		delta_out = np.multiply( -(error), self.derivative_fnct(self.L3)) #change of layer 3
		output_weight = np.dot(np.transpose(self.A2) ,delta_out) * lr

		delta_hidden = np.dot(delta_out, np.transpose(self.W2)) * self.derivative_fnct(self.L2) #change of layer 2
		hidden_weight = np.dot(np.transpose(input_data), delta_hidden)

		return hidden_weight, output_weight, delta_hidden, delta_out #first two are the derivates for input to the gradient. 



	def cost_function(self, expected_output, task = "SSE"):
		'''Calculate cost of descent via loss function. 

		| This function addresses how we minimize our cost function, J. It is called by the fit() function.
		| For all of our expected training values, calculate the error between true (y) and our current predictons (yhat)

		Parameters: 
			expected_output (np array): expected output of network. 
			task (str): default "SSE". Selected loss function. Options are "SSE": sum of squared errors, "MSE": mean squared error. (and hopefully binary cross entropy coming soon. )
		Returns:
			error (float)
		'''
	
		if task == "SSE":
			error = 0
			#for all our true labels, calculate the different between y and yhat, square this an sum it across all errors. 
			for i in range(expected_output.shape[0]):
				for j in range(expected_output.shape[1]):
					error += (expected_output[i,j] - self.yhat[i,j]) ** 2
			return error

		elif task == "MSE":
			val = []
			for i in range(expected_output.shape[0]):
				for j in range(expected_output.shape[1]):
					error = (expected_output[i,j] - self.yhat[i,j]) ** 2
					val.append(error)
			return np.mean(val)
		
		else:
		 	print("Amanda, haven't added others"); exit()

	def update_weights(self, dW1, dW2, db1, db2, learning_rate, m):
		'''Update layer weights. 

		| This function will calculate the new weights and bias values given the current neuron deltas and our optimizer parameters (so far just learning rate, I didn't add momentum)

		Parameters: 
			dW1 (np array): gradient of hidden weights
			dW2 (np array): gradient of output weights
			db1 (np array): gradient of hidden biases
			db2 (np array): gradient of output biases
			learning_rate (float): how should we traverse this gradient
			m (int): number of entries in our input (aka 1/m in front of the summations)
			
		Returns:
			error (float)
		'''

		#amanda in your previoius version you had:   a = neuron.weights[j] + -self.learn_rate * neuron.delta * neuron.weights[j]
		
		#I've simplified this a bunch now but it comes from 
		#vector = vector -learn_rate * gradient(vector)
		#where gradient(vector) = ((1/m) * vector) * vector
		self.W1 = self.W1 - learning_rate*(((1/m) * dW1) + 0 * self.W1)
		self.W2 = self.W2 - learning_rate*(((1/m) * dW2) + 0 * self.W2)
		
		self.bias1 = self.bias1 - learning_rate*((1/m) * db1)
		self.bias2 = self.bias2 - learning_rate*((1/m) * db2)

		return(self)


	def init_matrices_to_zero(self):
		'''Initialize gradients to zero. 

		| At the begining of every new epoch in our gradient descent we need to start from zeros. This is a helper function to create those initial matrices. 

		Returns:
			W1, W2, b1, b2 (np arrays) : all np arrays of zeros in differening sizes. 
		'''

		W1 = np.zeros((self.input_size, self.hidden_size))
		b1 = np.zeros((1,self.hidden_size))
		W2 = np.zeros((self.hidden_size, self.output_size))
		b2 = np.zeros((1,self.output_size))
		return(W1, W2, b1, b2)


	def fit(self, input_data, expected_output, epochs=10, learning_rate=0.01, loss_function="SSE", cc=0.0001, verbose=False):
		'''Fit/Train the model with inputs and hyperparameters. 

		| Train the model based on the input data to learn the expected output. The function employs batch gradient descent as an optimizer. 
		| Training will stop when number of epochs is reached or change in error between epochs is less than cc

		Parameters: 
			input_data (np array): input data to flow through network. 
			expected_output (np array): expected final output (classes in NN or identical to input_data in autoencoder)
			epochs (int): default 10. Number of epochs to run in batch gradient descent
			learning_rate (float): default 0.01. Learning rate of gradient descent
			loss_function (str): default "SSE". Choice of loss function to optimizer; current options are sum of squared error (SSE) and mean squared error (MSE)
			cc (float): default 0.0001 -- convergence criteria. When change in error is less than this value, stop iterating. 
			verbose (bool): default False. If True, will iteratively print the error per epoch. 
		'''

		#initialze inputs
		self.fit_statistics = []
		self.input_data = input_data
		self.expected_output = expected_output
		n_entries = input_data.shape[0]  #number of rows in input dataframe
				# aka "m" in formal gradient descent equation apparently so I switched to that in cost functions. 
		
		i = 0
		not_converged = True
		quitting_time = cc
		
		while not_converged:
			# For every iteration, start our gradients at zero. 
			dW1, dW2, db1, db2 = self.init_matrices_to_zero()

			#Feed information forward to get new predicted ouputs
			#Then let that information flow back and calculate the errors (delta)
			self.yhat = self.feedforward(input_data)
			hidden_weight, output_weight, delta_hidden, delta_out = self.backprop(input_data, expected_output)
			
			#For all the gradients, update them with our new values. 
			#For the matrices this is simple addition. 
			dW1 += hidden_weight
			dW2 += output_weight

			#However, for vectors we have to iteratively add them 
			for sample in range(0,(n_entries-1)):
				db1 += delta_hidden[sample]
				db2 += delta_out[sample]

			#Calculate the error in this particular epoch and save it to our final statistics dataframe (used for plotting later)
			er = self.cost_function(expected_output, loss_function)
			self.fit_statistics.append([i, er])
			#or, optionally print to screen if the user wants. 
			if verbose:
				print('>epoch=%d, error=%.3f' % (i, er))
				
			#Use the mean gradients to update our parameters. 
			self.update_weights(dW1, dW2, db1, db2, learning_rate, n_entries)


			#Check to see if we're done
			i+=1
			if i == epochs:
				#print("quitting bc epochs")
				not_converged = False

			if i > 1:
				if abs(er - self.fit_statistics[-2][1]) <= quitting_time:
					print("Reached convergence at epoch: ", i)
					not_converged = False

		#Calculate final score of model. 
		self.final_score = self.cost_function(expected_output, loss_function)
		return 

	def predict(self, input_data, task=None):
		'''Predict output layer for new inputs

		| With the provided input data, what does the model predict as the output.

		Parameters: 
			input_data (np array): input data to flow through network. This should be new to network and not the training data ideally. 
			task (str): default None. If None, will return the actual output layer predictions. If "round", will return the values rounded to the nearest integer. This is helpful in binary class predictions. 
		Returns:
			outputs (np array): output predictions in a matrix the same size as input_data
		'''

		outputs = self.feedforward(input_data)
		if (task == "round"):
			return np.round(outputs, decimals=0) #arg max
		else:
			return outputs









if __name__ == "__main__":

	#sorry im putting my main here again. I know its annoying but I find it way easier to test this way. 

	amanda_auto  = NeuralNetwork(setup_nodes = (8, 3, 8), 
								activation = "sigmoid", 
								seed=1)

	x = np.identity(8)
	y = x
	amanda_auto.fit(x, y, epochs=1000, learning_rate=20, loss_function ="MSE", cc =0, verbose=True)
	print(amanda_auto.final_score)
	predict = amanda_auto.predict(x, task="round")
	print("PREDICT", predict)
	exit()



