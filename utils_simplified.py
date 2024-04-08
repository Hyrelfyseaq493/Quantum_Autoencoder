import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import h5py
from typing import Tuple, Callable, Dict, NamedTuple, List
from pennylane import numpy as qnp
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import sys
import yaml
import os
from pathlib import Path
from pennylane import numpy as qnp
from scipy.special import rel_entr, kl_div
import matplotlib.pyplot as plt
import pennylane as qml
from circuits import ansatz_dict, embedding_dict

"""Modules for reading configs and saving models"""


#############################################################################
class ConfigReader:
    def __init__(self, file_path: str = None):
        """_summary_

        Args:
            file_path (str, optional): _description_. Defaults to None.
        """
        self.file_path = file_path

    def config_file(self) -> argparse.ArgumentParser:
        """_summary_

        Returns:
            argparse.ArgumentParser: _description_
        """

        parser = argparse.ArgumentParser(
            description="which config to run",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--config", type=str, default="configs/batch_size_experiment.yaml"
        )
        return parser

    def config_override(self) -> argparse.ArgumentParser:
        """_summary_

        Returns:
            argparse.ArgumentParse: _description_
        """
        parser = argparse.ArgumentParser(
            description="parameters to override config",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("--override", type=bool, default=False)
        parser.add_argument("--dataset", type=str, default="higgs")
        parser.add_argument("--ansatz", type=str, default="ansatz_1")
        parser.add_argument("--embedding", type=str, default="rx_embedding")
        parser.add_argument("--loss_fn", type=str, default="fidelity_loss")
        parser.add_argument("--latent_size", type=int, default="3")
        parser.add_argument("--input_size", type=int, default="9")
        parser.add_argument("--train_size", type=int, default="200")
        parser.add_argument("--normal_test_size", type=int, default="1000")
        parser.add_argument("--anomaly_test_size", type=int, default="100")
        parser.add_argument("--split", type=list, default=[0.5, 0.5])
        parser.add_argument("--layers", type=int, default=1)
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--batch_size", type=int, default=10)
        parser.add_argument("--w_init_dist", type=str, default="uniform")
        parser.add_argument("--w_init_range", type=list, default=[0, 2 * np.pi])
        parser.add_argument("--lr", type=float, default=0.01)
        parser.add_argument("--optimiser", type=str, default="Adam")
        parser.add_argument("--optimiser_backend", type=str, default="autograd")
        parser.add_argument("--seed", type=int, default=0)

        return parser

    def parse_args(self, args, parser):
        all_args = parser.parse_known_args(args)[0]
        return all_args

    def read_config(self, config_yaml):
        with open(config_yaml, "r") as f:
            data = yaml.load(f, Loader=yaml.UnsafeLoader)
        return data

    def display_config(self, config, title):
        print("#" * 62)
        print("#" + f"{title}".center(60, " ") + "#")
        for k, v in config.items():
            entry = f"{k}: {v}".center(60, " ")
            print(f"#{entry}#")
        print("#" * 62)

    def compile_config(self):
        if self.file_path:
            config_file_path = self.file_path
        else:
            parser_config = self.config_file()
            args_config = self.parse_args(sys.argv[1:], parser_config)
            config_file_path = args_config.config

        config = self.read_config(config_file_path)

        parser_override = self.config_override()
        args_override = self.parse_args(sys.argv[1:], parser_override)

        passed_cmds = sys.argv
        if "--config" in passed_cmds:
            passed_cmds.remove("--config")

        override_vars = [s[2:] for s in passed_cmds[1:] if "--" in s]
        if args_override.override == True:
            for i in range(len(override_vars)):
                new_value = getattr(args_override, override_vars[i])
                config[override_vars[i]] = new_value

        np.random.seed(config["seed"])
        qnp.random.seed(config["seed"])
        general_config = self.compile_general_config(config)
        classical_config = self.compile_classical_config(config)
        quantum_config = self.compile_quantum_config(config)
        print(classical_config)
        return general_config, classical_config, quantum_config

    def compile_general_config(self, config):
        general_config = config.copy()
        general_config.pop("quantum")
        general_config.pop("classical")
        self.display_config(general_config, "SETUP PARAMETERS FOR TRAINING")

        return general_config

    def compile_quantum_config(self, config):
        quantum_config = config["quantum"]

        quantum_config["trash_size"] = (
            quantum_config["input_size"] - quantum_config["latent_size"]
        )
        quantum_config["total_wires"] = (
            quantum_config["input_size"] + quantum_config["trash_size"] + 1
        )

        self.display_config(quantum_config, "QUANTUM CONFIG")

        quantum_config["ansatz_fn"] = ansatz_dict[quantum_config["ansatz"]]
        quantum_config["embedding_fn"] = embedding_dict[quantum_config["embedding"]]

        return quantum_config

    def compile_classical_config(self, config):
        classical_config = config["classical"]
        self.display_config(classical_config, "CLASSICAL CONFIG")
        return classical_config


def setup_save_dir(config, path=None):
    if path:
        s = path

    else:
        s = (
            "qmodel_"
            + config["loss_fn"]
            + "_"
            + config["embedding"]
            + "_"
            + config["ansatz"]
        )

    rootdir = Path(
        os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
        + "/anomaly_detection"
        + "/saved_models/"
        + config["dataset"]
        + f"/{s}"
    )

    if not rootdir.exists():
        os.makedirs(str(rootdir))

    subdir, dirs, files = next(os.walk(rootdir))

    run_index = f"run_{len(dirs)+1}"
    save_dir = (
        Path(
            os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
            + "/anomaly_detection"
            + "/saved_models/"
            + config["dataset"]
        )
        / s
        / run_index
    )
    print(save_dir)
    if not save_dir.exists():
        os.makedirs(str(save_dir))

    # parameters_file = open(f"{str(save_dir)}/parameters.yaml", "w")
    # yaml.dump(config, parameters_file)
    # parameters_file.close()

    return save_dir


def convert_h5_to_lists(h5_file, config):
    experiment_losses = h5py.File(h5_file, "r")
    bg_loss, sg_loss, bgc_loss, sgc_loss = [], [], [], []
    key = list(config["experiment_var"].keys())[0]
    var_list_len = len(config["experiment_var"][key])
    var_set_len = len(set(config["experiment_var"][key]))
    if var_set_len == var_list_len:
        for _, var in enumerate(config["experiment_var"][key]):
            bg_trainsize, sg_trainsize, bgc_trainsize, sgc_trainsize = [], [], [], []
            for k in range(config["k_folds"]):
                bgc_trainsize.append(
                    experiment_losses[f"classical/bg_losses/{key}_{var}/fold_{k}"][:]
                )
                sgc_trainsize.append(
                    experiment_losses[f"classical/sg_losses/{key}_{var}/fold_{k}"][:]
                )
                bg_trainsize.append(
                    experiment_losses[f"quantum/bg_losses/{key}_{var}/fold_{k}"][:]
                )
                sg_trainsize.append(
                    experiment_losses[f"quantum/sg_losses/{key}_{var}/fold_{k}"][:]
                )

            bg_loss.append(bg_trainsize)
            sg_loss.append(sg_trainsize)
            bgc_loss.append(bgc_trainsize)
            sgc_loss.append(sgc_trainsize)
    else:
        for idx, var in enumerate(config["experiment_var"][key]):
            new_var = var + str(idx)
            bg_trainsize, sg_trainsize, bgc_trainsize, sgc_trainsize = [], [], [], []
            for k in range(config["k_folds"]):
                bgc_trainsize.append(
                    experiment_losses[f"classical/bg_losses/{key}_{new_var}/fold_{k}"][
                        :
                    ]
                )
                sgc_trainsize.append(
                    experiment_losses[f"classical/sg_losses/{key}_{new_var}/fold_{k}"][
                        :
                    ]
                )
                bg_trainsize.append(
                    experiment_losses[f"quantum/bg_losses/{key}_{new_var}/fold_{k}"][:]
                )
                sg_trainsize.append(
                    experiment_losses[f"quantum/sg_losses/{key}_{new_var}/fold_{k}"][:]
                )

            bg_loss.append(bg_trainsize)
            sg_loss.append(sg_trainsize)
            bgc_loss.append(bgc_trainsize)
            sgc_loss.append(sgc_trainsize)

    return bg_loss, sg_loss, bgc_loss, sgc_loss


#############################################################################


frame_stack = []


@dataclass
class Frame:
    params: Dict[str, qnp.ndarray]
    is_initialising: bool = False


def current_frame():
    return frame_stack[-1]


class Transformed(NamedTuple):
    init: Callable
    apply: Callable


def transform(f) -> Transformed:
    def init_f(*args, **kwargs):
        frame_stack.append(Frame({}, is_initialising=True))
        f(*args, **kwargs)
        frame = frame_stack.pop()
        return frame.params

    def apply_f(params, *args, **kwargs):
        frame_stack.append(Frame(params))
        outs = f(*args, **kwargs)
        frame_stack.pop()
        return outs

    return Transformed(init_f, apply_f)


def get_param(identifier, shape):
    frame = current_frame()
    if frame.is_initialising:
        frame.params[identifier] = qnp.random.normal(size=shape, requires_grad=True)
    return frame.params[identifier]


def haar_dist(n, num_bins):
    bins_list = []
    for i in range(num_bins + 1):
        bins_list.append((i) / num_bins)
    # Center of the Bean
    bins_x = []
    for i in range(num_bins):
        bins_x.append(bins_list[1] + bins_list[i])

    def P_harr(l, u, N):
        return (1 - l) ** (N - 1) - (1 - u) ** (N - 1)

    # Harr historgram
    P_harr_hist = []
    for i in range(num_bins):
        P_harr_hist.append(P_harr(bins_list[i], bins_list[i + 1], 2**n))

    return bins_x, bins_list, P_harr_hist


def expressivity(circuit, param_shape, n, layers=1, sample_size=200, bins=200):
    param_shape.insert(0, sample_size)
    shape = tuple(param_shape)
    params = qnp.random.uniform(0, 2 * np.pi, size=shape)
    samples = []
    fidelities = np.zeros(sample_size)
    for i in range(sample_size):
        sample = circuit(params[i], range(shape[2]))
        samples.append(sample)

    for i in range(sample_size):
        psi = samples[i]
        for j in range(sample_size):
            if i != j:
                phi = samples[j]
                f = qml.math.fidelity(psi, phi)
                fidelities[i] = f

    bins_x, bins_list, P_haar_hist = haar_dist(n, bins)
    weights = np.ones_like(fidelities) / float(fidelities.shape[0])
    fids, _, _ = plt.hist(
        fidelities, bins=bins_list, weights=weights, range=[0, 1], label="params"
    )
    plt.close()
    # plt.plot(bins_x, P_haar_hist)
    # plt.legend()
    # plt.show()

    kl = kl_div(fids, P_haar_hist)  # rel_entr(fids, P_haar_hist)

    return np.sum(kl)


def mayer_wallach(circuit, layers, n_qubits, size, sample=1024):
    res = np.zeros(sample, dtype=complex)
    N = n_qubits

    for i in range(sample):
        params = np.random.uniform(0, 2 * np.pi, size=size)
        entropy = 0

        for j in range(N):
            dens = circuit(params, layers, wire=j, wires=range(N))
            trace = np.trace(dens**2)
            entropy += trace

        entropy /= N
        res[i] = 1 - entropy

    return 2 * np.sum(res).real / sample


def dynamic_mayer_wallach(circuit, n_qubits, p):
    N = n_qubits

    entropy = 0
    for j in range(N):
        dens = circuit(wire=j, param=p)
        trace = np.trace(dens**2)
        entropy += trace

    entropy /= N
    res = 1 - entropy

    return 2 * res.real


class DynamicEntanglement:
    def __init__(self, circuit_properties):
        self.ansatz = circuit_properties["ansatz_fn"]
        self.n_qubits = circuit_properties["input_size"]
        self.circuit_properties = circuit_properties

    def dynamic_mayer_wallach(self, p):
        entropy = 0

        def circuit(params, wires):
            self.ansatz(params, range(params.shape[1]), self.circuit_properties)
            return qml.density_matrix(wires=wires)

        dev = qml.device("default.qubit", wires=self.n_qubits)
        matrix_gen = qml.QNode(circuit, dev)

        for j in range(self.n_qubits):
            dens = matrix_gen(params=p, wires=j)
            trace = np.trace(dens**2)
            entropy += trace

        entropy /= self.n_qubits
        res = 1 - entropy

        return 2 * res.real

    def __call__(self, params):
        return self.dynamic_mayer_wallach(params)

    def __str__(self):
        return "dynamic_entanglement"


class DynamicEntanglement:
    def __init__(self, circuit_properties):
        self.ansatz = circuit_properties["ansatz_fn"]
        self.n_qubits = circuit_properties["input_size"]
        self.circuit_properties = circuit_properties

    def entanglement_capability(self, params, n_shots=1000):
        n_shots = int(n_shots)

        def circuit(params):
            self.ansatz(params, range(params.shape[1]), self.circuit_properties)
            return qml.state()

        dev = qml.device("default.qubit", wires=self.n_qubits)
        circuit_fn = qml.QNode(circuit, dev)
        # estimate fidelities
        entanglements = []
        for _ in range(n_shots):
            rho = circuit_fn(params)
            entanglements.append(entanglement_measure(rho))
        return np.mean(np.array(entanglements))

    def __call__(self, params):
        return self.entanglement_capability(params)

    def __str__(self):
        return "dynamic_entanglement"


class EntanglementEntropy:
    def __init__(self, circuit_properties):
        self.ansatz = circuit_properties["ansatz_fn"]
        self.n_qubits = circuit_properties["input_size"]
        self.latent_size = circuit_properties["latent_size"]
        self.circuit_properties = circuit_properties

    def entanglement_entropy(self, p):
        def circuit(params):
            self.ansatz(params, range(params.shape[1]), self.circuit_properties)
            return qml.vn_entropy(wires=range(self.latent_size))

        dev = qml.device("default.qubit", wires=self.n_qubits)
        entropy_fn = qml.QNode(circuit, dev)
        entropy = entropy_fn(p)
        return entropy

    def __call__(self, params):
        return self.entanglement_entropy(params)

    def __str__(self):
        return "entanglement_entropy"


def entanglement_capability(
    circuit_simulator, n_params, config, wires, n_shots=1000, data=None, seed=None
):
    """Computes entanglement capability for a circuit.

    Args:
        circuit_simulator: A function that takes n_params and returns a qiskit
            Statevector or DensityMatrix.
        n_params: The number of parameters circuit_simulator accepts. Presumed
            to be uniformly distributed in [0, 2pi]
        n_shots: How often the simulation should be repeated.
        data: Array of data for the case of data-based expressibility computation. The
              values of the circuit parameters are sampled from the data distribution
              instaed of uniformly from [0, 2pi].

    Returns:
        The expressiblity of the circuit.
    """
    if seed is not None:
        np.random.seed(seed)
    n_shots = int(n_shots)

    # estimate fidelities
    entanglements = []
    for _ in range(n_shots):
        if data is not None:
            params = data[np.random.choice(data.shape[0], size=1)].flatten()
        else:
            params = np.random.uniform(0, 2 * np.pi, size=n_params)
        rho = circuit_simulator(params, wires, config)
        entanglements.append(entanglement_measure(rho))
    return (
        np.mean(np.array(entanglements)),
        np.std(np.array(entanglements)),
        np.min(np.array(entanglements)),
        np.max(np.array(entanglements)),
        entanglements,
    )


def entanglement_measure(rho):
    """Calculates Meyer and Wallach's entanglement measure Q.

    See https://arxiv.org/abs/quant-ph/0108104 for more.

    Args:
        rho: qiskit Statevector (or convertable) representation of the state to
            calculate Q of.

    Returns:
        Q_value: The Meyer-Wallach entanglement measure of density_matrix.
    """
    n_qubits = int(np.log2(rho.shape[0]))
    entanglement_sum = 0

    rho_data = rho  # rho.data
    for k in range(n_qubits):
        # Elements of the statevector for which the kth qubit is 0/1 respectively
        k_zero_mask = 0 == np.arange(2**n_qubits) // 2**k % 2
        k_one_mask = 1 == np.arange(2**n_qubits) // 2**k % 2

        rho_k_zero = rho_data[k_zero_mask]
        rho_k_one = rho_data[k_one_mask]

        entanglement_sum += wedge_distance(rho_k_zero, rho_k_one)

    return 4 / n_qubits * entanglement_sum


def wedge_distance(u, v):
    """Calculates the wedge distance between input vectors u and v.

    Args:
        u: Vector 1
        v: Vector 2

    Returns:
        Wedge product of u and v.

    Remarks:
        Could be more efficient, but realistically speaking this function is
        not the bottleneck of the entanglement capability calculation.
    """
    n_it = np.size(u)
    sum = 0
    for i in range(1, n_it):
        for j in range(i):
            sum += np.abs(u[i] * v[j] - u[j] * v[i]) ** 2
    return sum


"""
class DataLoader:
    def __init__(self):
        pass

    def higgs(self, train_size, normal_test_size, anomaly_test_size, split, scale):
        train_bg = pd.read_csv("data/higgs_dataset/train/train_sample_bg_morestat.csv")
        test_bg = pd.read_csv("data/higgs_dataset/test/test_sample_bg_morestat.csv")
        test_sg = pd.read_csv("data/higgs_dataset/test/sample_mH2500_morestat.csv")
        train_bg.drop("Unnamed: 0", axis=1, inplace=True)
        test_bg.drop("Unnamed: 0", axis=1, inplace=True)
        test_sg.drop("Unnamed: 0", axis=1, inplace=True)

        scaler = StandardScaler()
        train_bg = scaler.fit_transform(train_bg)
        test_bg = scaler.transform(test_bg)
        test_sg = scaler.transform(test_sg)

        training_data = train_bg[:train_size]
        test_normal = test_bg[:normal_test_size]
        test_anomaly = test_sg[:anomaly_test_size]

        np.random.shuffle(training_data)
        split0_index = int(split[0] * training_data.shape[0])
        split1_index = int((split[0] + split[1]) * training_data.shape[0])

        train_data = training_data[:split0_index]
        validation_data = training_data[split0_index:split1_index]
        background_labels = np.zeros(test_normal.shape[0])
        signal_labels = np.ones(test_anomaly.shape[0])
        test_labels = np.concatenate((background_labels, signal_labels))
        test_data = np.vstack((test_normal, test_anomaly))

        p = np.random.permutation(test_labels.shape[0])
        test_labels = test_labels[p]
        test_data = test_data[p]

        return train_data, validation_data, test_data, test_labels

    def graviton(self, args):
        print("Signal ", end="")
        x_sig = self.h5_to_ml_ready_numpy(args["sig_path"])
        print("Background ", end="")
        x_bkg = self.h5_to_ml_ready_numpy(args["bkg_path"])
        print("Testing background ", end="")
        x_bkg_test = self.h5_to_ml_ready_numpy(args["test_bkg_path"])

        try:
            train_loader = self.get_train_dataset(
                x_sig, x_bkg, args["ntrain"], args["unsup"]
            )
        except KeyError:
            print(
                "No training dataset is loaded, since no"
                " args['ntrain'] was provided."
            )
            train_loader = None

        test_loader = self.get_test_dataset(
            x_sig, x_bkg_test, args["ntest"]
        )  # , args["unsup"])
        return train_loader, test_loader

    def h5_to_ml_ready_numpy(self, file_path):
        h5_file = h5py.File(file_path, "r")
        latent_rep = np.asarray(h5_file.get("latent_space"))
        latent_rep_flat = self.reshaper(latent_rep)
        return latent_rep_flat

    def reshaper(self, array):
        print(f"reshape {array.shape}", end="")
        array = np.reshape(array, (len(array), -1))
        print(f" -> {array.shape}")
        return array

    def get_train_dataset(
        self,
        sig: np.ndarray,
        bkg: np.ndarray,
        ntrain: int,
        is_unsup: bool,
    ):
        if is_unsup:
            x_data_train = bkg[:ntrain]
            print(
                f"Created training dataset of shape: {x_data_train.shape}, "
                "for unsupervised training."
            )
            return x_data_train, None

        sig = sig[-int(ntrain / 2) :]
        bkg = bkg[: int(ntrain / 2)]
        x_data_train = np.concatenate((sig, bkg))
        print(
            f"Created training dataset of shape: {x_data_train.shape} "
            "for supervised training."
        )
        y_data_train = self.create_output_y(len(sig))
        return x_data_train, y_data_train

    def get_test_dataset(
        self,
        sig: np.ndarray,
        bkg_test: np.ndarray,
        ntest: int,
        is_unsup: bool = False,
    ):
        if is_unsup:
            sig = sig[: int(ntest * 0.05)]
            bkg_test = bkg_test[: int(ntest * 0.95)]
            y_data_test = self.create_output_y(n=(len(sig), len(bkg_test)))
            print(f"Signal shape {sig.shape} and Background shape {bkg_test.shape}")
        else:
            sig = sig[: int(ntest / 2)]
            bkg_test = bkg_test[: int(ntest / 2)]
            y_data_test = self.create_output_y(int(ntest / 2))
            print(f"Signal shape {sig.shape} and Background shape {bkg_test.shape}")

        x_data_test = np.concatenate((sig, bkg_test))
        print(f"Created testing dataset of shape: {x_data_test.shape}")
        return x_data_test, y_data_test

    def create_output_y(self, n):

        if isinstance(n, Tuple):
            n_sig, n_bkg = n[0], n[1]
            y_data = np.concatenate((np.ones(n_sig), np.zeros(n_bkg)))
            print(y_data.shape)
            return y_data
        y_data = np.concatenate((np.ones(n), np.zeros(n)))
        return y_data
"""


class HiggsDataset(Dataset):
    """Higgs Dataset"""

    def __init__(
        self,
        root_dir,
        partition="train",
        scale=True,
        transform=None,
    ):
        self.root_dir = root_dir
        self.partition = partition
        self.transform = transform

        self.train_bg = pd.read_csv(f"{root_dir}train/train_sample_bg_morestat.csv")
        self.test_bg = pd.read_csv(f"{root_dir}test/test_sample_bg_morestat.csv")
        self.test_sg = pd.read_csv(f"{root_dir}test/sample_mH2500_morestat.csv")
        self.train_bg.drop("Unnamed: 0", axis=1, inplace=True)
        self.test_bg.drop("Unnamed: 0", axis=1, inplace=True)
        self.test_sg.drop("Unnamed: 0", axis=1, inplace=True)

        max_values = {
            "lep1pt": 1000,
            "lep2pt": 900,
            "theta_ll": np.pi,
            "b1pt": 1000,
            "b2pt": 900,
            "theta_bb": np.pi,
            "MET": 1000,
            "dR_l1": 5.12484,
            "dR_l2": 4.95733,
        }
        min_values = {
            "lep1pt": 0.0,
            "lep2pt": 0.0,
            "theta_ll": 0.0,
            "b1pt": 0.0,
            "b2pt": 0.0,
            "theta_bb": 0.0,
            "MET": 0.0,
            "dR_l1": 0.494697,
            "dR_l2": 0.493649,
        }
        keys = min_values.keys()
        max_array = np.array(list(max_values.values()))
        min_array = np.array(list(min_values.values()))

        self.train_bg = self.train_bg[keys].values
        self.test_bg = self.test_bg[keys].values
        self.test_sg = self.test_sg[keys].values

        if scale:
            # self.scaler = StandardScaler()
            # self.scaler.fit(self.train_bg)
            # self.train_bg = self.scaler.transform(self.train_bg)
            # self.test_bg = self.scaler.transform(self.test_bg)
            # self.test_sg = self.scaler.transform(self.test_sg)

            self.train_bg, _ = scale_feature(
                self.train_bg, max_array=max_array, min_array=min_array
            )
            self.test_bg, _ = scale_feature(
                self.test_bg, max_array=max_array, min_array=min_array
            )
            self.test_sg, _ = scale_feature(
                self.test_sg, max_array=max_array, min_array=min_array
            )

        if self.partition == "train":
            self.data = self.train_bg
            self.labels = np.zeros(self.data.shape[0])

        if self.partition == "test":
            self.test_bg = self.test_bg
            self.test_sg = self.test_sg
            self.data = np.vstack((self.test_bg, self.test_sg))

            self.bg_labels = np.zeros(self.test_bg.shape[0])
            self.sg_labels = np.ones(self.test_sg.shape[0])
            self.labels = np.concatenate((self.bg_labels, self.sg_labels)).flatten()

    def __str__(self):
        return "higgs"

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)
        return qnp.array(sample, requires_grad=False), label

    def get_test_chunk(self, bg_size, sg_size):
        self.test_bg, self.test_sg = self.test_bg[:bg_size], self.test_sg[:sg_size]
        np.random.shuffle(self.test_bg)
        np.random.shuffle(self.test_sg)
        test_data = np.vstack((self.test_bg, self.test_sg))
        test_labels = np.concatenate((np.zeros(bg_size), np.ones(sg_size)))

        if self.transform:
            test_data = self.transform(test_data)
        return test_data, test_labels

    def split(self, train_size, validation_size):
        dataset_size = self.train_bg.shape[0]
        train_idx = np.random.choice(range(dataset_size), train_size, replace=False)
        remaining_idxs = np.array(list(set(range(dataset_size)) - set(train_idx)))
        val_idx = np.random.choice(remaining_idxs, size=validation_size, replace=False)
        return train_idx, val_idx


def scale_feature(X, epsilon=0.0, max_array=None, min_array=None):
    """Takes in an array of features and scales them between 0 and pi
        removes datapoints which exceed have a feature which exceeds the
        the maximum set for the given feature in max_array.
        X is the scaled features with the first row being the min_array
        and the final row being the max array. Passed_inds is an array
        of indices for datapoints which were included in X.

    Args:
        X (_type_): _description_
        epsilon (float, optional): _description_. Defaults to 0.0.
        max_array (_type_, optional): _description_. Defaults to None.
        min_array (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    scaler = MinMaxScaler(feature_range=(0, 1- epsilon))
    #scaler = MinMaxScaler(feature_range=(0, np.pi - epsilon))
    if max_array is None:
        X = scaler.fit_transform(X)
        passed = np.arange(len(X))
    else:
        assert min_array is not None
        passed_inds = []
        for array, M in zip(np.swapaxes(X, 0, 1), max_array):
            passed_inds.append(array < M)
        passed_inds = np.array(passed_inds)
        passed_inds = np.sum(passed_inds, axis=0)
        rejected = np.where(passed_inds != X.shape[-1])[0]
        passed_inds = passed_inds == X.shape[-1]
        X = X[passed_inds]
        X = np.concatenate(([min_array], X, [max_array]), axis=0)
        X = scaler.fit_transform(X)[1:-1]
    return X, passed_inds


class Scale(object):
    def __init__(self, train_bg):
        self.scaler = StandardScaler()
        self.scaler.fit(train_bg)

    def __call__(self, sample):
        return self.scaler.transform(sample)


class Permute(object):
    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, sample):
        if len(sample.shape) == 2:
            return sample[:, self.permutation]
        else:
            return sample[self.permutation]


class dataset(Dataset):
    """A -> HZ -> ZZZ"""

    def __init__(
        self,
        root_dir,
        partition="train",
        latent_rep=4,
        scale=True,
        transform=None,
        validation_size=500,
    ):
        self.root_dir = root_dir
        self.partition = partition
        self.validation_size = validation_size
        self.transform = transform

        self.test_sg1 = self.h5_to_ml_ready_numpy(
            f"{self.root_dir}/latent{latent_rep}/sg1.h5"
        )[:15000,:]
        self.test_sg2 = self.h5_to_ml_ready_numpy(
            f"{self.root_dir}/latent{latent_rep}/sg2.h5"
        )[:15000,:]

        self.test_bg = self.h5_to_ml_ready_numpy(
            f"{self.root_dir}/latent{latent_rep}/bg.h5"
        )[:15000,:]

        if scale:
            # self.scaler = StandardScaler()
            # self.scaler.fit(self.train_bg)
            # self.train_bg = self.scaler.transform(self.train_bg)
            # self.test_bg = self.scaler.transform(self.test_bg)
            # self.test_sg = self.scaler.transform(self.test_sg)
            max_array = np.ones(2 * latent_rep)
            min_array = np.ones(2 * latent_rep) * -1

            self.test_bg, _ = scale_feature(
                self.test_bg, max_array=max_array, min_array=min_array
            )
            self.test_sg1, _ = scale_feature(
                self.test_sg1, max_array=max_array, min_array=min_array
            )
            self.test_sg2, _ = scale_feature(
                self.test_sg2, max_array=max_array, min_array=min_array
            )

        if self.partition == "test":
        #    self.test_bg = self.test_bg
        #    self.test_sg = self.test_sg
        #    self.data = np.vstack((self.test_bg, self.test_sg))

            self.bg_labels = np.zeros(self.test_bg.shape[0])
            self.sg1_labels = np.ones(self.test_sg1.shape[0])
            self.sg2_labels = np.ones(self.test_sg2.shape[0])
        #    self.labels = np.concatenate((self.bg_labels, self.sg_labels)).flatten()

    def h5_to_ml_ready_numpy(self, file_path):
        h5_file = h5py.File(file_path, "r")
        latent_rep = np.asarray(h5_file.get("latent_space"))
        latent_rep_flat = self.reshaper(latent_rep)
        return latent_rep_flat

    def reshaper(self, array):
        print(f"reshape {array.shape}", end="")
        array = np.reshape(array, (len(array), -1))
        print(f" -> {array.shape}")
        return array

    def __str__(self):
        return "zzz"

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        bg_sample = self.test_bg[idx]
        sg1_sample = self.test_sg1[idx]
        sg2_sample = self.test_sg2[idx]
        bg_label = self.bg_labels[idx]
        sg1_label = self.sg1_labels[idx]
        sg2_label = self.sg2_labels[idx]

        if self.transform:
            bg_sample = self.transform(bg_sample)
            sg1_sample = self.transform(sg1_sample)
            sg2_sample = self.transform(sg2_sample)
        return (qnp.array(bg_sample, requires_grad=False), 
                qnp.array(sg1_sample, requires_grad=False),
                qnp.array(sg2_sample, requires_grad=False),
                bg_label, sg1_label, sg2_label)

    def get_test_chunk(self, bg_size, sg_size):
        bg_size = int(bg_size)
        sg_size = int(sg_size)
        bg_ids = np.random.choice(
            range(int(self.test_bg.shape[0])), bg_size, replace=False
        )
        sg_ids = np.random.choice(
            range(int(self.test_sg.shape[0])), sg_size, replace=False
        )
        test_data = np.vstack((self.test_bg[bg_ids], self.test_sg[sg_ids]))
        test_labels = np.concatenate((np.zeros(bg_size), np.ones(sg_size)))
        if self.transform:
            test_data = self.transform(test_data)
        return test_data, test_labels

    def split(self, train_size, validation_size):
        dataset_size = self.train_bg.shape[0]
        train_idx = np.random.choice(
            range(int(dataset_size)), train_size, replace=False
        )
        remaining_idxs = np.array(list(set(range(int(dataset_size))) - set(train_idx)))
        val_idx = np.random.choice(remaining_idxs, size=validation_size, replace=False)
        return train_idx, val_idx


class NarrowGDataset(Dataset):
    """A -> HZ -> ZZZ"""

    def __init__(
        self,
        root_dir,
        partition="train",
        latent_rep=4,
        scale=True,
        transform=None,
        validation_size=500,
    ):
        self.root_dir = root_dir
        self.partition = partition
        self.validation_size = validation_size
        self.transform = transform

        self.train_bg = self.h5_to_ml_ready_numpy(
            f"{self.root_dir}/latent{latent_rep}/latentrep_train_QCD_sig.h5"
        )

        self.test_sg = self.h5_to_ml_ready_numpy(
            f"{self.root_dir}/latent{latent_rep}/latentrep_RSGraviton_WW_NA_35.h5"
        )

        self.test_bg = self.h5_to_ml_ready_numpy(
            f"{self.root_dir}/latent{latent_rep}/latentrep_test_QCD_sig.h5"
        )

        if scale:
            # self.scaler = StandardScaler()
            # self.scaler.fit(self.train_bg)
            # self.train_bg = self.scaler.transform(self.train_bg)
            # self.test_bg = self.scaler.transform(self.test_bg)
            # self.test_sg = self.scaler.transform(self.test_sg)

            max_array = np.ones(2 * latent_rep)
            min_array = np.ones(2 * latent_rep) * -1

            self.test_bg, _ = scale_feature(
                self.test_bg, max_array=max_array, min_array=min_array
            )
            self.test_sg, _ = scale_feature(
                self.test_sg, max_array=max_array, min_array=min_array
            )

        if self.partition == "train":
            self.data = self.train_bg
            self.labels = np.zeros(self.data.shape[0])

        if self.partition == "test":
            self.test_bg = self.test_bg
            self.test_sg = self.test_sg
            self.data = np.vstack((self.test_bg, self.test_sg))

            self.bg_labels = np.zeros(self.test_bg.shape[0])
            self.sg_labels = np.ones(self.test_sg.shape[0])
            self.labels = np.concatenate((self.bg_labels, self.sg_labels)).flatten()

    def h5_to_ml_ready_numpy(self, file_path):
        h5_file = h5py.File(file_path, "r")
        latent_rep = np.asarray(h5_file.get("latent_space"))
        latent_rep_flat = self.reshaper(latent_rep)
        return latent_rep_flat

    def reshaper(self, array):
        print(f"reshape {array.shape}", end="")
        array = np.reshape(array, (len(array), -1))
        print(f" -> {array.shape}")
        return array

    def __str__(self):
        return "ng"

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)
        return qnp.array(sample, requires_grad=False), label

    def get_test_chunk(self, bg_size, sg_size):
        bg_size = int(bg_size)
        sg_size = int(sg_size)
        bg_ids = np.random.choice(
            range(int(self.test_bg.shape[0])), bg_size, replace=False
        )
        sg_ids = np.random.choice(
            range(int(self.test_sg.shape[0])), sg_size, replace=False
        )
        test_data = np.vstack((self.test_bg[bg_ids], self.test_sg[sg_ids]))
        test_labels = np.concatenate((np.zeros(bg_size), np.ones(sg_size)))
        if self.transform:
            test_data = self.transform(test_data)
        return test_data, test_labels

    def split(self, train_size, validation_size):
        dataset_size = self.train_bg.shape[0]
        train_idx = np.random.choice(range(dataset_size), train_size, replace=False)
        remaining_idxs = np.array(list(set(range(dataset_size)) - set(train_idx)))
        val_idx = np.random.choice(remaining_idxs, size=validation_size, replace=False)
        return train_idx, val_idx


class WideGDataset(Dataset):
    """A -> HZ -> ZZZ"""

    def __init__(
        self,
        root_dir,
        partition="train",
        latent_rep=4,
        scale=True,
        transform=None,
        validation_size=500,
    ):
        self.root_dir = root_dir
        self.partition = partition
        self.validation_size = validation_size
        self.transform = transform

        self.train_bg = self.h5_to_ml_ready_numpy(
            f"{self.root_dir}/latent{latent_rep}/latentrep_train_QCD_sig.h5"
        )

        self.test_sg = self.h5_to_ml_ready_numpy(
            f"{self.root_dir}/latent{latent_rep}/latentrep_RSGraviton_WW_BR_15.h5"
        )

        self.test_bg = self.h5_to_ml_ready_numpy(
            f"{self.root_dir}/latent{latent_rep}/latentrep_test_QCD_sig.h5"
        )

        if scale:
            max_array = np.ones(2 * latent_rep)
            min_array = np.ones(2 * latent_rep) * -1

            self.train_bg, _ = scale_feature(
                self.train_bg, max_array=max_array, min_array=min_array
            )
            self.test_bg, _ = scale_feature(
                self.test_bg, max_array=max_array, min_array=min_array
            )
            self.test_sg, _ = scale_feature(
                self.test_sg, max_array=max_array, min_array=min_array
            )

        if self.partition == "train":
            self.data = self.train_bg
            self.labels = np.zeros(self.data.shape[0])

        if self.partition == "test":
            self.test_bg = self.test_bg
            self.test_sg = self.test_sg
            self.data = np.vstack((self.test_bg, self.test_sg))

            self.bg_labels = np.zeros(self.test_bg.shape[0])
            self.sg_labels = np.ones(self.test_sg.shape[0])
            self.labels = np.concatenate((self.bg_labels, self.sg_labels)).flatten()

    def h5_to_ml_ready_numpy(self, file_path):
        h5_file = h5py.File(file_path, "r")
        latent_rep = np.asarray(h5_file.get("latent_space"))
        latent_rep_flat = self.reshaper(latent_rep)
        return latent_rep_flat

    def reshaper(self, array):
        print(f"reshape {array.shape}", end="")
        array = np.reshape(array, (len(array), -1))
        print(f" -> {array.shape}")
        return array

    def __str__(self):
        return "wg"

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)
        return qnp.array(sample, requires_grad=False), label

    def get_test_chunk(self, bg_size, sg_size):
        bg_size = int(bg_size)
        sg_size = int(sg_size)
        bg_ids = np.random.choice(
            range(int(self.test_bg.shape[0])), bg_size, replace=False
        )
        sg_ids = np.random.choice(
            range(int(self.test_sg.shape[0])), sg_size, replace=False
        )
        test_data = np.vstack((self.test_bg[bg_ids], self.test_sg[sg_ids]))
        test_labels = np.concatenate((np.zeros(bg_size), np.ones(sg_size)))
        if self.transform:
            test_data = self.transform(test_data)
        return test_data, test_labels

    def split(self, train_size, validation_size):
        dataset_size = self.train_bg.shape[0]
        train_idx = np.random.choice(range(dataset_size), train_size, replace=False)
        remaining_idxs = np.array(list(set(range(dataset_size)) - set(train_idx)))
        val_idx = np.random.choice(remaining_idxs, size=validation_size, replace=False)
        return train_idx, val_idx


class GaussianDataset(Dataset):
    """A -> HZ -> ZZZ"""

    def __init__(
        self,
        root_dir,
        partition="train",
        latent_rep=4,
        scale=True,
        transform=None,
        validation_size=500,
    ):
        self.root_dir = root_dir
        self.partition = partition
        self.validation_size = validation_size
        self.transform = transform

        self.train_bg = self.h5_to_ml_ready_numpy(
            f"{self.root_dir}/latent{latent_rep}/latentrep_train_QCD_sig.h5"
        )

        self.test_sg = self.h5_to_ml_ready_numpy(
            f"{self.root_dir}/latent{latent_rep}/latentrep_RSGraviton_WW_BR_15.h5"
        )

        self.test_bg = self.h5_to_ml_ready_numpy(
            f"{self.root_dir}/latent{latent_rep}/latentrep_test_QCD_sig.h5"
        )

        if scale:
            # self.scaler = StandardScaler()
            # self.scaler.fit(self.train_bg)
            # self.train_bg = self.scaler.transform(self.train_bg)
            # self.test_bg = self.scaler.transform(self.test_bg)
            # self.test_sg = self.scaler.transform(self.test_sg)

            max_array = np.ones(2 * latent_rep)
            min_array = np.ones(2 * latent_rep) * -1

            self.train_bg, _ = scale_feature(
                self.train_bg, max_array=max_array, min_array=min_array
            )
            self.test_bg, _ = scale_feature(
                self.test_bg, max_array=max_array, min_array=min_array
            )
            self.test_sg, _ = scale_feature(
                self.test_sg, max_array=max_array, min_array=min_array
            )

        if self.partition == "train":
            self.data = self.train_bg
            self.labels = np.zeros(self.data.shape[0])

        if self.partition == "test":
            self.test_bg = self.test_bg
            self.test_sg = self.test_sg
            self.data = np.vstack((self.test_bg, self.test_sg))

            self.bg_labels = np.zeros(self.test_bg.shape[0])
            self.sg_labels = np.ones(self.test_sg.shape[0])
            self.labels = np.concatenate((self.bg_labels, self.sg_labels)).flatten()

    def h5_to_ml_ready_numpy(self, file_path):
        h5_file = h5py.File(file_path, "r")
        latent_rep = np.asarray(h5_file.get("latent_space"))
        latent_rep_flat = self.reshaper(latent_rep)
        return latent_rep_flat

    def reshaper(self, array):
        print(f"reshape {array.shape}", end="")
        array = np.reshape(array, (len(array), -1))
        print(f" -> {array.shape}")
        return array

    def __str__(self):
        return "wg"

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)
        return qnp.array(sample, requires_grad=False), label

    def get_test_chunk(self, bg_size, sg_size):
        bg_size = int(bg_size)
        sg_size = int(sg_size)
        bg_ids = np.random.choice(
            range(int(self.test_bg.shape[0])), bg_size, replace=False
        )
        sg_ids = np.random.choice(
            range(int(self.test_sg.shape[0])), sg_size, replace=False
        )
        test_data = np.vstack((self.test_bg[bg_ids], self.test_sg[sg_ids]))
        test_labels = np.concatenate((np.zeros(bg_size), np.ones(sg_size)))
        if self.transform:
            test_data = self.transform(test_data)
        return test_data, test_labels

    def split(self, train_size, validation_size):
        dataset_size = self.train_bg.shape[0]
        train_idx = np.random.choice(range(dataset_size), train_size, replace=False)
        remaining_idxs = np.array(list(set(range(dataset_size)) - set(train_idx)))
        val_idx = np.random.choice(remaining_idxs, size=validation_size, replace=False)
        return train_idx, val_idx


################


def weight_init(min, max, distribution, shape):
    distribution_class = getattr(qnp.random, distribution)
    init_params = distribution_class(min, max, size=(shape), requires_grad=True)
    return init_params


###################


def get_fpr_around_tpr_point(fpr, tpr, tpr_working_point=0.8):
    ind = np.array([])
    low_bound = tpr_working_point * 0.999
    up_bound = tpr_working_point * 1.001

    while len(ind) == 0:
        ind = np.where(np.logical_and(tpr >= low_bound, tpr <= up_bound))[0]
        low_bound *= 0.99  # open the window by 1%
        up_bound *= 1.01
    fpr_window_no_zeros = fpr[ind][fpr[ind] != 0]
    one_over_fpr_mean = np.mean(1.0 / fpr_window_no_zeros), np.std(
        1.0 / fpr_window_no_zeros
    )
    print(
        f"\nTPR values around {tpr_working_point} window with lower bound {low_bound}"
        f" and upper bound: {up_bound}"
    )
    print(
        f"Corresponding mean 1/FPR value in that window: {one_over_fpr_mean[0]:.3f} ± "
        f"{one_over_fpr_mean[1]:.3f}"
    )
    return one_over_fpr_mean


class DatasetFactory:
    def __init__(self):
        pass

    def create_dataset(self, dataset, **kwargs):
        print(dataset)
        serializer = self._get_dataset(dataset)
        return serializer(**kwargs)

    def _get_dataset(self, dataset):
        if dataset == "higgs":
            return HiggsDataset
        elif dataset == "zzz":
            return ZZZDataset
        elif dataset == "ng":
            return NarrowGDataset
        elif dataset == "wg":
            return WideGDataset
        else:
            raise ValueError(dataset)


"""
f = h5py.File("data/latent_reps/latent8/latentrep_train_QCD_sig.h5", "r")
print(list(f.keys()))
print(f.name)
dset = f["latent_space"]
print(dset.dtype)
print(dset.shape)


f = h5py.File("data/latent_reps/latent8/latentrep_QCD_sig.h5", "r")
print(list(f.keys()))
print(f.name)
dset = f["latent_space"]
print(dset[0, 1, :])
print(dset.dtype)
print(dset.shape)


f = h5py.File("data/latent_reps/latent32/latentrep_QCD_sig.h5", "r")
print(list(f.keys()))
print(f.name)
dset = f["latent_space"]
print(dset.dtype)
print(dset.shape)
test_subset = dset[: int(1436672 / 2)]
train_subset = dset[int(1436672 / 2) :]
print(test_subset.shape)
print(train_subset.shape)

new_qcd_train = h5py.File("data/latent_reps/latent32/latentrep_train_QCD_sig.h5", "w")
train_dset = new_qcd_train.create_dataset(
    "latent_space", train_subset.shape, data=train_subset, dtype=np.float32
)


new_qcd_train = h5py.File("data/latent_reps/latent32/latentrep_test_QCD_sig.h5", "w")
test_dset = new_qcd_train.create_dataset(
    "latent_space", test_subset.shape, data=test_subset, dtype=np.float32
)
"""
