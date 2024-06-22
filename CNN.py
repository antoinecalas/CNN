from pickle import NONE
import numpy as np
import random
from mnist import MNIST

def relu(x):
    #print("relu used")
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReluDerived(x):
    output =[]
    for value in x:
        if value>0:
            output.append(1)
        else :
            output.append(0)
    return np.array([output])


def initialize_weights(shape):
    
    return np.random.normal(0, 1, size=shape)

def softmax(z):
    #print("softmax used")
    # Subtract the max value from z for numerical stability
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z, axis=-1, keepdims=True)
    return exp_z / sum_exp_z

class ConvLayer:
    def __init__(self,nb_filters,size,nb_channels,input_image_size,index):
        self.index = index
        self.size = size
        self.nb_filter = nb_filters
        self.output_image_size = input_image_size-2
        self.output = []
        self.filters = initialize_weights((nb_filters,nb_channels,size,size))
        
    def GetOutputChannels(self):
        return len(self.filters)
    
    def __str__(self):
        x = f"┌————————————————————┑\n|   conv layer n°{self.index}   |\n|          {self.size}*{self.size}       |\n|      {self.nb_filter} filters     |\n|____________________|\n"
        return(x)
    
    def GetOutputSize(self):
        #print("cov output: ",self.output_image_size*self.output_image_size)
        return self.output_image_size*self.output_image_size
    
    def ComputeLayer(self,input):
        try :
            channels,height,width = np.shape(input)
            #print(channels,height,width)
        except:
            height,width = np.shape(input)
            channels = 1

        output = np.zeros((self.nb_filter,height-2,width-2))

        for i_filter,filter in enumerate(self.filters) :
            for y in range(1,height-1):
                for x in range(1,width-1):
                    for z in range(channels):
                        top_left_value = input[z][y-1][x-1] * filter[z][0][0]
                        top_center_value = input[z][y-1][x] * filter[z][0][1]
                        top_right_value = input[z][y-1][x+1] * filter[z][0][2]
                        middle_left_value = input[z][y][x-1] * filter[z][1][0]
                        middle_center_value = input[z][y][x] * filter[z][1][1]
                        middle_right_value = input[z][y][x+1] * filter[z][1][2]
                        bottom_left_value = input[z][y+1][x-1] * filter[z][2][0]
                        bottom_center_value = input[z][y+1][x] * filter[z][2][1]
                        bottom_right_value = input[z][y+1][x+1] * filter[z][2][2]
                        output[i_filter][y-1][x-1] += top_left_value + top_center_value +top_right_value+middle_left_value+middle_center_value+middle_right_value+bottom_left_value+bottom_center_value+bottom_right_value
                    output[i_filter][y-1][x-1] = output[i_filter][y-1][x-1]
            #print(output[i_filter],"\n")
        self.output = relu(output)
        print("conv layer computed")
        return self.output
    
class FullyConnectedLayer:
    def __init__(self,nb_neurons,input_size,index,activation_function):
        self.index = index
        self.activation_function = activation_function
        self.nb_neurons = nb_neurons
        self.output = []
        self.weights = initialize_weights((nb_neurons,input_size))

    def __str__(self):
        x = "                 "
        x = x[self.nb_neurons:]
        y = x[self.nb_neurons:]
        for i in range(self.nb_neurons) :
            x += "O "
            y += f"|{self.weights[i][0]:.2f}"
            
        print(x)
        print(y)
        return("")

    def GetOutputSize(self):
        #print("fc output: ",self.nb_neurons)
        return self.nb_neurons


    def ComputeLayer(self,inputs):#------------------------------------------------------need to add a func and a bias
        #print(inputs)
        output = np.zeros(self.nb_neurons)
        for j, weights in enumerate(self.weights):
            #print(weights,"  ")
            for i, weight in enumerate(weights):
            
                output[j] += inputs[i]*weight
        #print(self.activation_function,"FC output :\n untouched : ",output,"\n after fct :",self.activation_function(output))
        
        self.output = self.activation_function(output)
        #print("FC computed : ")
        return self.output
    
    def UpdateWeights(self,prior_activation,error,learning_rate):
        #print("-------------------------------------------------------------")
        #print("_____self output______________\n ",self.output)
        #print("_______errror input_______\n ",error)
        #print("_______prior activation_____\n",prior_activation)

        #if self.activation_function == relu :
        #    print(ReluDerived(self.output))
        #    layer_error = np.dot(error,ReluDerived(self.output))
        #elif self.activation_function == softmax:
        layer_error = error
        print("weight error : ",layer_error.T)
        

        layer_gradient = np.dot(layer_error,[prior_activation])
        #print("           !!!gradient !!!\n",layer_gradient,"\n")
        #print("\n           !!!self.weights !!!\n",self.weights)
        self.weights = self.weights - learning_rate*layer_gradient
        #print("\n           !!!self.weights !!!\n",self.weights)
        next_layer_error = np.dot(self.weights.T,layer_error)
        #print("\n           !!!next layer error !!!\n",next_layer_error)
        return next_layer_error


class FlattenLayer:
    def __init__(self,input_size,index):
        self.index = index
        self.size = input_size
    
    def __str__(slef):
        print("             |flatten|")
        return("")
    
    def GetOutputSize(self):
        
        return self.size
    
    def ComputeLayer(self,input):
        #print("flatten computed")
        self.output = input.flatten()
        return self.output
    
    def UpdateWeights(self,prior_activation,error,learning_rate):
        return error

class CNN:
    def __init__(self,show_weights_flag,image_size):
        self.model = []
        self.FCLayers_matrix = []
        self.show_weights_flag = show_weights_flag
        self.image_size = image_size
    
    def __str__(self):
        print("_____________________________________\n")
        for layer in self.model :
            print(layer)
        print("_____________________________________\n")
        return ""    

    def AddFullyConnectedLayer(self,nb_neurons,activation_function):
        if self.model:
            self.model.append(FullyConnectedLayer(nb_neurons,self.model[-1].GetOutputSize(),len(self.model),activation_function))
            print(f"--- Fully connected layer ({nb_neurons} neurons)added succesfully with input {nb_neurons,self.model[-2].GetOutputSize()}---")
        else : 
            self.model.append(FullyConnectedLayer(nb_neurons,self.image_size*self.image_size,len(self.model),activation_function))
            print(f"--- Fully connected layer ({nb_neurons} neurons)added succesfully with input {self.image_size}---")
        
    def AddConvLayer(self,size,nb_filter):
        if self.model:
            self.model.append(ConvLayer(nb_filter,size,self.model[-1].GetOutputChannels(),self.model[-1].GetOutputSize(),len(self.model)))
        else :
            self.model.append(ConvLayer(nb_filter,size,1,self.image_size,len(self.model)))
        print(f"--- {nb_filter} filters, {size}*{size} Convolutionnal layer added succesfully---")

    def AddFlattenLayer(self):
        if self.model:
            print(self.model[-1].GetOutputSize())
            self.model.append(FlattenLayer(self.model[-1].GetOutputSize(),len(self.model)))
        else :
            self.model.append(FlattenLayer(self.image_size*self.image_size,len(self.model)))
        print("--- Flatten layer added succesfully---")

    def Run(self,input_layer):
        current_layer_value = input_layer

        for layer in self.model:
            current_layer_value = layer.ComputeLayer(current_layer_value)
            print(current_layer_value)

        #print("\n\n output :\n",current_layer_value)
        return current_layer_value

    def Train(self,training_data,nb_epoch,batch_size = 1,learning_rate = 0.01):

        mndata = MNIST(training_data)
        images, labels = mndata.load_training()
        index = random.randrange(0, len(images))  # choose an index ;-)
        expected_output = np.zeros(10)
        nb_layers = len(self.model)
        

        for epoch in range(nb_epoch):
            error_term = np.zeros(3)
            for i_batch in range(batch_size):
                index_training = random.randrange(0, len(images))
                #expected_output[labels[index_training]] = 1
                
                #output = self.Run(np.reshape(images[index_training],(28,28)))
                output = self.Run(np.array([[2,5]]))
                #print(output)
                expected_output = [1,0,0]
                
                #print(output,expected_output)
                u = (output-expected_output)*2
                error_term += u
                
                #expected_output[labels[index_training]] = 0
                
            
                
            error_term = np.around(np.array([error_term/batch_size]).T,4)
            print(f"output : {np.around(output,2)}  error  : {np.around(error_term,2)}")
            for i in range(1,nb_layers):
                index = nb_layers-i
                #print(index,self.model[index])
                #print("output of layer = ", self.model[index].output)
                error_term = self.model[index].UpdateWeights(self.model[index-1].output,error_term,learning_rate)    
            self.model[0].UpdateWeights(training_data,error_term,learning_rate)





                
                   




    
    
