import mpmath
import numpy as np

from pynever import nodes, tensors
from pynever.networks import SequentialNetwork
from pynever.strategies.verification.algorithms import SSBPVerification
from pynever.strategies.verification.parameters import SSBPVerificationParameters
from pynever.strategies.verification.properties import VnnLibProperty

# NETWORK DEFINITION
# W = np.array([[1, 1], [1, -1]])
# W2 = np.array([[1, 1], [0, 1]])
# b2 = np.array([1, 0])
# W = np.array([[0.6, -0.4], [0.25, 0.75]])
W = mpmath.matrix([[0.60, -0.40], [0.25, 0.75]])
W2 = np.eye(2, 2)
b2 = np.zeros(2)

fc_1 = nodes.FullyConnectedNode('FC_1', (2,), 2, W, tensors.zeros((2, )))
rl_1 = nodes.ReLUNode('ReLU_1', (2,))

nn = SequentialNetwork('NN', 'X')
nn.append_node(fc_1)
nn.append_node(rl_1)

prop = VnnLibProperty('2d_prop.vnnlib')
print(prop)

# print(SSBPVerification(SSBPVerificationParameters()).verify(nn, prop))
