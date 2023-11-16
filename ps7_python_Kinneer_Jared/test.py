import numpy as np
from numpy.random import rand

#dataset
x = np.array(([0.9,0.8],[0.6,0.3],[0.9,0.1],[0.9,0.8]))  #Features
y = np.array(([0],[1],[1],[0]))  #Labels (0,1)

#Activation Function
def Sigmoid(z):
    return  1/ (1 + np.exp(-z)) # The Sigmoid Function 

class NeuralNetwork:
    # Step one
    def __init__(self,x,y,nodes_first_layer = 6 , nodes_second_layer = 4, nodes_output_layer = 1):
        # Define x,y
        self.inputs_of_layer0 = x
        self.y = y
        
        # Define number of neurns in each layer
        self.nodes_first_layer = nodes_first_layer
        self.nodes_second_layer = nodes_second_layer
        self.nodes_output_layer = nodes_output_layer
        
        #intialize the wieghts (theta) metrices
        
        self.thetas_of_layer0 = np.random.rand(self.inputs_of_layer0.shape[1] + 1, self.nodes_first_layer) #shape: [2+1, 6]
        print("layer0\n",self.thetas_of_layer0)
        self.thetas_of_layer1 = np.random.rand(self.nodes_first_layer + 1, self.nodes_second_layer) #shape: [6 + 1, 4]
        self.thetas_of_layer2 = np.random.rand(self.nodes_second_layer + 1,self.nodes_output_layer) #shape: [4 + 1, 1]
        
    # Step Two
    def FeedForward(self):
        #compute all the nodes (a1, a2, a3, a4, a5, a6) in layer1
        print("layer0[0]\n",self.thetas_of_layer0[0])
        print("inputs\n",self.inputs_of_layer0)
        print("layer0[1:]\n",self.thetas_of_layer0[1:])
        self.Z1 = self.thetas_of_layer0[0] + np.dot(self.inputs_of_layer0, self.thetas_of_layer0[1:] )
        self.layer1 = Sigmoid(self.Z1)
        print("layer1\n",self.layer1)
        #compute all the nodes (a1, a2, a3, a4) in layer2
        self.Z2 = self.thetas_of_layer1[0] + np.dot(self.layer1, self.thetas_of_layer1[1:])
        self.layer2 = Sigmoid(self.Z2)
        
        #compute the nodes (a1) in layer3
        self.Z3 = self.thetas_of_layer2[0] + np.dot(self.layer2, self.thetas_of_layer2[1:])
        self.layer3 = Sigmoid(self.Z3) #Output layer
        
        return self.layer3

NN = NeuralNetwork(x,y)

predicted_output = NN.FeedForward()
    
print ("Actual Output: \n", y)
print("Predicted Output: \n", predicted_output, "\n")