from pickle import NONE
import numpy as np
import random

class CNN:
    def __init__(self,show_weights_flag):
        self.weight_matrix = []
        self.FCLayers_matrix = []
        self.show_weights_flag = show_weights_flag
    
    def __str__(self):
        shift = "                              "
        printed_model = "\n"

        

        if self.show_weights_flag == True:
            for layer in self.weight_matrix:
                print(len(np.shape(layer)))
                if len(np.shape(layer))==2:
                    print("entered a fc layer \n")
                    for weigths in layer :
                        print(weigths)
                if len(np.shape(layer))==3:
                    print("entered the first conv layer \n")
                    for weigths in layer :
                        print(weigths)
                if len(np.shape(layer))==4:
                    print("entered  conv layer \n")
                    for weigths in layer :
                        print(weigths)
                    
                    """ for nb_neurons in self.FCLayers_matrix:
                        printed_model += shift[nb_neurons:]
                        for i in range(nb_neurons) :
                            printed_model +="O "
                        printed_model +="\n\n"


                nb_element = int((np.size(layer)*2+3*len(layer))/2)
                printed_model += shift[nb_element:]
                for weigths in layer :
                    printed_model +="| "
                    for weight in weigths:
                        printed_model += str(int(weight)) +" "
                    printed_model +="|"
                printed_model +="\n" """
            printed_model +="\n"

        return printed_model

    def ComputeNeuron(self,inputs,weights):#------------------------------------------------------need to add a func and a bias
        output = 0
        for input, weight in zip(inputs,weights):
            output += input*weight
        return output

    def AddFullyConnectedLayer(self,nb_neurons):
        if self.FCLayers_matrix:
            x = np.random.randint(-3, 3, (nb_neurons,self.FCLayers_matrix[-1]), int)
            self.weight_matrix.append(x)
        
        self.FCLayers_matrix.append(nb_neurons)
        print(f"--- Fully connected layer ({nb_neurons} neurons)added succesfully---")

    def AddConvLayer(self,size,nb_filter):
        if self.weight_matrix:
            print("nb channels :",len(self.weight_matrix[-1]))
            x = np.random.randint(-3,3,(nb_filter,len(self.weight_matrix[-1]),size,size), int)
            self.weight_matrix.append(x)
        else :
            x = np.random.randint(-3,3,(nb_filter,size,size), int)
        
        self.weight_matrix.append(x)

        print(f"--- {nb_filter} filters, {np.shape(x)} Convolutionnal layer added succesfully---")


    def Run(self,input):
        #print(self.weight_matrix)
        width  = input.shape[0]
        self.weight_matrix.append(np.random.randint(-3, 3, (self.FCLayers_matrix[0],width), int))

        

        last_layer_value = input
        current_layer_value = []

        for layer in self.weight_matrix:
            
            for i, weights_for_neuron in enumerate(layer) :
                current_layer_value.append(self.ComputeNeuron(last_layer_value,weights_for_neuron))

            #print(current_layer_value)
            last_layer_value = current_layer_value
            current_layer_value = []
        


                
                   




    
    
