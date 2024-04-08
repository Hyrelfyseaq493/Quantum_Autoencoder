import argparse
import pennylane as qml
from pennylane import numpy as np
import time
import os
import json
from model import CircuitSearchModel, NAS_search_space
from evolution.evolution_sampler import EvolutionSampler
import random
from utils_simplified import dataset, Permute
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser("QAS")
parser.add_argument('--test_result_save', type=str, default='testing_result_bsmb_0.33_with_loss', help='experiment name')
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=500, help='num of training epochs')
parser.add_argument('--warmup_epochs', type=int, default=50, help='num of warmup epochs')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='', help='which architecture to use')
parser.add_argument('--noise', action='store_true', default=False, help='use noise')
# circuit
parser.add_argument('--n_qubits', type=int, default=6, help='number of qubits')
parser.add_argument('--n_encode_layers', type=int, default=1, help='number of encoder layers')
parser.add_argument('--n_layers', type=int, default=3, help='number of layers')
# QAS
parser.add_argument('--n_experts', type=int, default=5, help='number of experts')
parser.add_argument('--n_search', type=int, default=500, help='number of search')
parser.add_argument('--searcher', type=str, default='random', help='searcher type', choices=['random', 'evolution'])
parser.add_argument('--ea_pop_size', type=int, default=25, help='population size of evolutionary algorithm')
parser.add_argument('--ea_gens', type=int, default=20, help='generation number of evolutionary algorithm')


args = parser.parse_args()
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(args.save)
with open(os.path.join(args.save, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

if args.noise:
    import qiskit
    import qiskit.providers.aer.noise as noise


def cost_fn(params, model, subnet, inputs, labels, records):
    loss = 0
    correct = 0
    size = inputs.shape[0]
    model.params = params
    for data, label in zip(inputs, labels):
        out = model(data, subnet)
        loss += (label - out)**2
        if (out < 0.5 and label == 0) or (out > 0.5 and label == 1):
            correct += 1
    loss /= size
    print('subnet: {}, loss: {}, acc: {}'.format(subnet, loss._value, correct / size))
    records['train_acc'].append(correct / size)
    records['loss'].append(loss._value)
    return loss


def test_fn(model, subnet, expert_idx, inputs, labels):
    correct = 0
    for data, label in zip(inputs, labels):
        model.params = model.params_space[expert_idx]
        out = model(data, subnet)

        if (out < 0.5 and label == 0) or (out > 0.5 and label == 1):
            correct += 1
    acc = correct / inputs.shape[0]
    return acc

def test_fn(model, subnet, expert_idx, inputs, labels):
    bg_loss = []
    sg_loss = []
    for feat, label in zip(inputs, labels):
        model.params = model.params_space[expert_idx]
        out = model(feat, subnet)
        if label == 1:
            sg_loss.append(out)
        elif label == 0:
            bg_loss.append(out)
        else:
            print("error")
    return bg_loss, sg_loss

def tpr_fpr(model, subnet, expert_idx, inputs, labels):
    loss = []
    for feat in inputs:
        model.params = model.params_space[expert_idx]
        out = model(feat, subnet)
        loss.append(out)
    loss = np.array(loss)
    auc_score = roc_auc_score(labels, loss)
    return auc_score


def expert_evaluator(model, subnet, inputs, labels, n_experts):
    r''' In this function, we locate the expert that achieves the minimum loss, where such an expert is the best choice
     for the given subset'''
    target_expert = 0
    target_loss = -1
    for i in range(n_experts):
        temp_loss = 0
        model.params = model.params_space[i]
        for data, label in zip(inputs, labels):
            out = model(data, subnet)
            temp_loss += (label - out)**2
        temp_loss /= inputs.shape[0]
        if target_loss == -1 or temp_loss < target_loss:
            target_loss = temp_loss
            target_expert = i
    return target_expert


def main():

    np.random.seed(args.seed)
    records = {
        'loss': [],
        'train_acc': [],
        'valid_acc': [],
        'test_acc': 0
    }
    '''init device'''
    if args.noise:
        # Error probabilities
        prob_1 = 0.05  # 1-qubit gate
        prob_2 = 0.2   # 2-qubit gate
        # Depolarizing quantum errors
        error_1 = noise.depolarizing_error(prob_1, 1)
        error_2 = noise.depolarizing_error(prob_2, 2)
        # Add errors to noise model
        noise_model = noise.NoiseModel()
        noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx'])
        print(noise_model)
        dev = qml.device('qiskit.aer', wires=args.n_qubits, noise_model=noise_model)
    else:
        dev = qml.device("default.qubit", wires=args.n_qubits)

    '''init model'''
    model = CircuitSearchModel(dev, args.n_qubits, args.n_encode_layers, args.n_layers, args.n_experts)
    opt = qml.AdamOptimizer(0.05)

    '''load mnist data'''
    data = dataset(
            "/Users/hyrelfyseaq493/qas_vqc",
            partition="test",
            scale=True,
            transform=Permute([6, 3, 0, 7, 2, 4]),
        )

    bg_data = data[:1000][0]
    sg1_data = data[:1000][1]
    sg2_data = data[:1000][2]
    bg_label = data[:1000][3]
    sg1_label = data[:1000][4]
    sg2_label = data[:1000][5]
    data_train = np.vstack((bg_data[:333],sg1_data[:333],sg2_data[:333]))
    label_train = np.concatenate((bg_label[:333],sg1_label[:333],sg2_label[:333])).flatten()
    data_valid = np.vstack((bg_data[333:666],sg1_data[333:666],sg2_data[333:666]))
    label_valid = np.concatenate((bg_label[333:666],sg1_label[333:666],sg2_label[333:666])).flatten()
    data_test1 = np.vstack((bg_data[666:], sg1_data[666:]))
    label_test1 = np.concatenate((bg_label[666:], sg1_label[666:])).flatten()
    data_test2 = np.vstack((bg_data[666:], sg2_data[666:]))
    label_test2 = np.concatenate((bg_label[666:], sg2_label[666:])).flatten()
    train_data_shape = data_train.shape
    test_data1_shape = data_test1.shape
    test_data2_shape = data_test2.shape
    print('train data size: {}'.format(train_data_shape))
    print('test data 1 size: {}'.format(test_data1_shape))
    print('test data 2 size: {}'.format(test_data2_shape))
    # randomize order of data
    random_ind1 = random.sample(range(test_data1_shape[0]), test_data1_shape[0])
    data_test1 = data_test1[random_ind1]
    label_test1 = label_test1[random_ind1]
    random_ind2 = random.sample(range(test_data2_shape[0]), test_data2_shape[0])
    data_test2 = data_test2[random_ind2]
    label_test2 = label_test2[random_ind2]
    # split data

    '''train'''
    for epoch in range(args.epochs):
        subnet = np.random.randint(0, len(NAS_search_space), (args.n_layers,)).tolist()
        # find the expert with minimal loss w.r.t. subnet
        if epoch < args.warmup_epochs:
            expert_idx = np.random.randint(args.n_experts)
        else:
            expert_idx = expert_evaluator(model, subnet, data_train, label_train, args.n_experts)
        print('{}/{}: subnet: {}, expert_idx: {}'.format(epoch,args.epochs,subnet, expert_idx))
        model.params_space[expert_idx] = opt.step(lambda params: cost_fn(params, model, subnet, data_train, label_train, records), model.params_space[expert_idx]) # train
        
    '''search'''
    print('Start search.')
    result = {}
    test_result = np.empty([args.n_search,5])
    if args.searcher == 'random':
        for i in range(args.n_search):
            subnet = np.random.randint(0, len(NAS_search_space), (args.n_layers,)).tolist()
            expert_idx1 = expert_evaluator(model, subnet, data_test1, label_test1, args.n_experts)
            bg_loss1, sg_loss1 = test_fn(model, subnet, expert_idx1, data_test1, label_test1)
            expert_idx2 = expert_evaluator(model, subnet, data_test2, label_test2, args.n_experts)
            bg_loss2, sg_loss2 = test_fn(model, subnet, expert_idx2, data_test2, label_test2)
    #@        result['-'.join([str(x) for x in subnet])] = acc
            print('{}/{}: subnet: {}, loss of sg1: {}, loss of sg2: {}'.format(i+1, args.n_search, subnet, np.mean(sg_loss1)-np.mean(bg_loss1), np.mean(sg_loss2)-np.mean(bg_loss2)))
            test_result[i,:-2] = subnet
            test_result[i,-2] = np.mean(sg_loss1)-np.mean(bg_loss1)
            test_result[i,-1] = np.mean(sg_loss2)-np.mean(bg_loss2)
    elif args.searcher == 'evolution':
        sampler = EvolutionSampler(pop_size=args.ea_pop_size, n_gens=args.ea_gens, n_layers=args.n_layers, n_blocks=len(NAS_search_space))
        def test_subnet_evolution(subnet):
            expert_idx = expert_evaluator(model, subnet, data_train, label_train, args.n_experts)
            acc = test_fn(model, subnet, expert_idx, data_valid, label_valid)
            return acc  # higher is better
        sorted_result = sampler.sample(test_subnet_evolution)
        result = sampler.subnet_eval_dict
    
    print('Search done.')
    np.savetxt(args.test_result_save, test_result)
 #   with open(os.path.join(args.save, 'nas_result.txt'), 'w') as f:
 #       f.write('\n'.join(['{} {}'.format(x[0], x[1]) for x in result.items()]))
 #  sorted_result = list(result.items())
 #   sorted_result.sort(key=lambda x: x[1], reverse=True)
 #   with open(os.path.join(args.save, 'nas_result_sorted.txt'), 'w') as f:
 #       f.write('\n'.join(['{} {}'.format(x[0], x[1]) for x in sorted_result]))
 #   '''save records'''
 #   json.dump(records, open(os.path.join(args.save, 'records.txt'), 'w'), indent=2)


if __name__ == '__main__':
    main()
