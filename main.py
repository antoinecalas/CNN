from CNN import *
import cv2



image = np.array([[[0,1,0,0,0],
         [0,1,0,0,0],
         [0,0,0,1,0],
         [0,0,0,1,0],
         [0,0,0,0,0]]])
image = cv2.imread("1.png",cv2.IMREAD_GRAYSCALE )
image = np.array(cv2.normalize(image,None,norm_type=cv2.NORM_MINMAX))
#print(image)
Cnn = CNN(True,1)
#Cnn.AddConvLayer(3,16)
#Cnn.AddConvLayer(3,3)
#Cnn.AddConvLayer(3,1,2)
Cnn.AddFlattenLayer()
Cnn.AddFullyConnectedLayer(2,sigmoid)
Cnn.AddFullyConnectedLayer(3,softmax)
print(Cnn)
input()
#Cnn.Run(image)
Cnn.Train("MNIST",200)

