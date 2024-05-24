from CNN import *

Cnn = CNN(True)
Cnn.AddConvLayer(3,2)
Cnn.AddConvLayer(3,4)
Cnn.AddConvLayer(3,1)
Cnn.AddFullyConnectedLayer(2)

Cnn.Run(np.array([1,2,0,0,3]))

print(Cnn)