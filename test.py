from mnist import MNIST
import random
import numpy as np

mndata = MNIST('MNIST')

images, labels = mndata.load_training()

index = random.randrange(0, len(images))  # choose an index ;-)
u = images[index]
#u = np.reshape(u,(28,28))
print(index,mndata.display(images[index]))