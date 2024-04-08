import pennylane as qml
from pennylane import numpy as np
import itertools
import numpy as onp

valid_Rs = [qml.RX, qml.RY, qml.RZ]
valid_CNOTs = ([0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5])

Rs_space = list(itertools.product(valid_Rs, valid_Rs, valid_Rs, valid_Rs, valid_Rs, valid_Rs))
CNOTs_space = [[y for y in CNOTs if y is not None] for CNOTs in list(itertools.product(*([x, None] for x in valid_CNOTs)))]
NAS_search_space = list(itertools.product(Rs_space, CNOTs_space))


def encode_layer(feature, j, n_qubits):
    '''
    encoder
    '''
    #qml.Hadamard(wires=0)
    #qml.Hadamard(wires=1)
    #qml.Hadamard(wires=2)
    for i in range(n_qubits):
        qml.RY(feature[i], wires=i)
    
    #phi = (np.pi - feature[0].val)*(np.pi - feature[1].val)*(np.pi - feature[2].val)
    #phi = (np.pi - feature[0])*(np.pi - feature[1])*(np.pi - feature[2])
    # CRY 1
    #qml.CRY(phi, wires=[0, 1])
    # CRY 2
    #qml.CRY(phi, wires=[1, 2])
    
    #phi = (np.pi - feature[0].val)*(np.pi - feature[1].val)*(np.pi - feature[2].val)

def layer(params, j, n_qubits):
    '''
    normal layer
    '''
    for i in range(n_qubits):
        qml.RY(params[j,  i], wires=i)
    #qml.CNOT(wires=[0, 1])
    #qml.CNOT(wires=[1, 2])


def qas_layer(params, j, n_qubits, Rs=[qml.RY, qml.RY, qml.RY], CNOTs=[[0, 1], [1, 2]], R_idx=None):
    for i in range(n_qubits):
        if R_idx is None:
            Rs[i](params[j, i], wires=i)
        else:
            Rs[i](params[j, R_idx, i], wires=i)
    for conn in CNOTs:
        qml.CNOT(wires=conn)
    for i in range(n_qubits/2):
        qml.CSWAP(wires=[3*n_qubits/2,n_qubits/2+i, n_qubits+i])

def swap_test(latent_size, trash_size):
    aux_qubit = latent_size + 2 * trash_size
    qml.Hadamard(wires=aux_qubit)
    for i in range(trash_size):
        qml.CSWAP(wires=[aux_qubit, latent_size + i, latent_size + trash_size + i])
    qml.Hadamard(wires=aux_qubit)

def qas_qa(params, n_layers, latent_size, trash_size, Rs, CNOTs):
    for j in range(n_layers):
        for i in range(latent_size+trash_size):
            Rs[j,i](params[j,i], wires=i)
    for j in CNOTs:
        qml.CNOT(wires=j)
    swap_test(latent_size, trash_size)



def circuit(feature, params, A=None, n_qubits=3, n_encode_layers=1, n_layers=3, arch=''):
    '''
    quantum circuit
    '''
    if arch == '':
        # normal circuit
        # repeatedly apply each layer in the circuit
        for j in range(n_encode_layers):
            encode_layer(feature, j, n_qubits)
        for j in range(n_layers):
            layer(params, j, n_qubits)
        return qml.expval(qml.Hermitian(A, wires=[0, 1, 2]))
    else:
        # nas circuit
        # repeatedly apply each layer in the circuit
        for j in range(n_encode_layers):
            encode_layer(feature, j, n_qubits)
        for j in range(n_layers):
            qas_layer(params, j, n_qubits, NAS_search_space[arch[j]][0], NAS_search_space[arch[j]][1])
        return qml.expval(qml.Hermitian(A, wires=[0, 1, 2]))


def circuit_decorator(dev, *args, **kwargs):
    return qml.qnode(dev)(circuit)(*args, **kwargs)


def circuit_search(feature, params, A=None, latent_size=3, trash_size=3, n_encode_layers=1, n_layers=3, arch=[]):
    '''
    quantum circuit
    '''
    # nas circuit
    # repeatedly apply each layer in the circuit
    #onp_params = onp.empty([*params.shape])
    #for i,j,k in zip(range(params.shape[0]),range(params.shape[1]),range(params.shape[2])):
    #    onp_params[i,j,k] = params[i,j,k]._value
    #print(type(params))
    for j in range(n_encode_layers):
        encode_layer(feature, j, latent_size + trash_size)
    qas_qa(params, n_layers, latent_size, trash_size, NAS_search_space[arch[j]][0], 
              NAS_search_space[arch[j]][1])
    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(latent_size+2*trash_size)]))

def circuit_search_decorator(dev, *args, **kwargs):
    #return qml.qnode(dev)(circuit_search)(*args, **kwargs)
    return qml.QNode(circuit_search,dev)(*args, **kwargs)

class CircuitModel():
    def __init__(self, dev, n_qubits=3, n_encode_layers=1, n_layers=3, arch=''):
        self.dev = dev
        self.n_qubits = n_qubits
        self.n_encode_layers = n_encode_layers
        self.n_layers = n_layers
        if arch != '':
            self.arch = [int(x) for x in arch.split('-')]
            assert len(self.arch) == n_layers
            print('----NAS circuit----')
            for i, idx in enumerate(self.arch):
                print('------layer {}-----'.format(i))
                print('Rs: {}'.format(NAS_search_space[idx][0]))
                print('CNOTs: {}'.format(NAS_search_space[idx][1]))
        else:
            self.arch = ''
        '''init params'''
        # randomly initialize parameters from a normal distribution
        self.params = np.random.uniform(0, np.pi * 2, (n_layers,   n_qubits))
        self.A = np.kron(np.eye(4), np.array([[1, 0], [0, 0]]))

    def __call__(self, x):
        out = circuit_decorator(self.dev, x, self.params,
                                A=self.A,
                                n_qubits=self.n_qubits, n_encode_layers=self.n_encode_layers,
                                n_layers=self.n_layers,
                                arch=self.arch)
        return out


class CircuitSearchModel():
    def __init__(self, dev, latent_size=3, trash_size=3, n_layers=3, n_experts=5):
        self.dev = dev
        self.latent_size = latent_size
        self.trash_size = trash_size
#        self.n_encode_layers = n_encode_layers
        self.n_layers = n_layers
        self.n_experts = n_experts
        '''init params'''
        # randomly initialize parameters from a normal distribution
        self.params_space = np.random.uniform(0, np.pi * 2, (n_experts, self.n_layers, self.latent_size + self.trash_size))
        self.params = None
        self.A = np.kron(np.eye(4), np.array([[1, 0], [0, 0]]))
    def __call__(self, feature, subnet: list):
        return qml.QNode(self.circuit_search,self.dev)(feature,
                                self.params,
                                A=self.A,
                                subnet = subnet)
    
    def circuit_search(self, feature, params, A=None, subnet=[]):
        '''
        quantum circuit
        '''
        self.encode_layer(feature)
        for j in range(self.n_layers):
            self.qas_encode_layer(j, Rs_space[subnet[j]])
        self.qas_CNOT_block(CNOTs_space[subnet[self.n_layers]])
        swap_test(self.latent_size, self.trash_size)
        #qas_qa(NAS_search_space[subnet[j]][0], NAS_search_space[subnet[j]][1])
        #return qml.expval(qml.Hermitian(A, wires=[0, 1, 2]))
        return qml.expval(qml.operation.Tensor(*[qml.PauliZ(self.latent_size + 2*self.trash_size)]))
    
    def encode_layer(self, feature):
        for i in range(self.latent_size + self.trash_size):
            qml.RY(feature[i], wires=i)

    def swap_test(self):
        aux_qubit = self.latent_size + 2 * self.trash_size
        qml.Hadamard(wires=aux_qubit)
        for i in range(self.trash_size):
            qml.CSWAP(wires=[aux_qubit, self.latent_size + i, self.latent_size + self.trash_size + i])
        qml.Hadamard(wires=aux_qubit)

    def qas_qa(self, Rs, CNOTs):
        for j in range(self.n_layers):
            for i in range(self.latent_size + self.trash_size):
                Rs[j][i](self.params[j,i], wires=i)
        for j in CNOTs:
            qml.CNOT(wires=j)
        swap_test(self.latent_size, self.trash_size)

    def qas_encode_layer(self,j,Rs):
        for i in range(self.latent_size + self.trash_size):
            Rs[i](self.params[j,i], wires=i)

    def qas_CNOT_block(self, CNOTs):
        for j in CNOTs:
            qml.CNOT(wires=j)