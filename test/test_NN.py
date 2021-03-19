from scripts import NN
import numpy as np

def test_general():
	amanda_auto  = NN.NeuralNetwork(setup_nodes = (8, 3, 8), activation = "sigmoid", seed=1)

	assert amanda_auto.W1.shape == (8, 3), "Fail 1"
	assert amanda_auto.W2.shape == (3, 8), "Fail 2"
	assert amanda_auto.bias1.shape == (1, 3), "Fail 3"
	assert amanda_auto.bias2.shape == (1, 8), "Fail 4"

	inputs = np.random.randint(2, size=(1,8))
	hidden_z, hidden_activation, output_z, output_activation = amanda_auto.feedforward(inputs, report = True)
	assert hidden_z.shape == (1, 3), "Fail 5"
	assert hidden_activation.shape == (1, 3), "Fail 6"
	assert output_z.shape == (1, 8), "Fail 7"
	assert output_activation.shape == (1, 8), "Fail 8"

	amanda_auto.yhat = output_z
	hidden_weight, output_weight, delta_hidden, delta_out = amanda_auto.backprop(inputs, inputs) #here they're the same thing. 
	assert hidden_weight.shape == (8, 3), "Fail 9"
	assert output_weight.shape == (3, 8), "Fail 10"
	assert delta_hidden.shape == (1, 3), "Fail 11"
	assert delta_out.shape == (1, 8), "Fail 12"


def test_encoder():
	amanda_auto  = NN.NeuralNetwork(setup_nodes = (8, 3, 8), activation = "sigmoid", seed=1)
	inputs = np.random.randint(2, size=(1,8))
	amanda_auto.fit(input_data=inputs, 
                expected_output = inputs, #here, training is the same as our hopeful prediction
                epochs = 10, 
                learning_rate = 20,
                lam = 0, 
                verbose=False) #dont want the output to be messy 
	assert amanda_auto.final_score < 0.1, "Fail 13"


def test_NN():
	X_train = np.asarray(
           [[2.7810836,2.550537003], [1.465489372,2.362125076], [3.396561688,4.400293529],
           [1.38807019,1.850220317],[3.06407232,3.005305973], [7.627531214,2.759262235],
           [5.332441248,2.088626775], [6.922596716,1.77106367],[8.675418651,-0.242068655],
           [7.673756466,3.508563011]]
        )

	y_train = np.asarray([[0], [0], [0],
                    [0], [0], [1],
                    [1], [1], [1], 
                    [1]])

	X_test = np.asarray([[1.235425, 2.121455],  # should be 0 with the lower scores
                    [6.1234, 2.1234]])  #should be 1 with higher scores
	#test comes from:
	#https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
            
	amanda_auto  = NN.NeuralNetwork(setup_nodes = (2, 3, 1), 
                             activation = "sigmoid", seed=1)
	amanda_auto.fit(input_data=X_train, 
                expected_output = y_train, #here, training is the same as our hopeful prediction
                epochs = 100, 
                learning_rate = 20,
                lam = 0, 
                verbose=False)
	predict = amanda_auto.predict(X_test, task="round")
	assert (predict == np.asarray([[0],[1]])).all() , "Fail 15"


def test_others():
	a = NN.oneHotDNA("AACGT", flatten = False)
	assert (a == np.array([[1., 0., 0., 0.],
       				   [1., 0., 0., 0.],
       				   [0., 1., 0., 0.],
       				   [0., 0., 1., 0.],
       				   [0., 0., 0., 1.]])).all(), "Failing to one-hot-encode"
	a = NN.oneHotDNA("AACGT", flatten = True)
	assert (a == np.array([1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.])).all(), "Failing to ecode + flatten"







