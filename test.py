import numpy as np

def softmax(z):
    z = z - np.max(z)  # For numerical stability
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z)
    return exp_z / sum_exp_z

# Given input and weights
input_data = np.array([-1268.18660981  -249.25866136])
weights = np.array([-0.25459539 -0.20320675])

# Compute the linear combination
z = np.dot(input_data, weights)

# Since softmax on a single value is 1, we'll assume a second neuron for a full example
# Assuming some second value, for example purposes let's use z2 = -10 for illustration
z2 = -10  # This is just an example value, not given in the problem

# Full softmax calculation
output = softmax(np.array([z, z2]))

print("Output with Softmax activation:", output)