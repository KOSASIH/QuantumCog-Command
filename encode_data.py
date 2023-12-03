from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap

def encode_data(data):
    num_features = len(data[0])
    num_qubits = num_features * 2  # We double the number of qubits for encoding
    
    qc = QuantumCircuit(num_qubits)
    
    feature_map = ZZFeatureMap(num_qubits, reps=1)
    qc.append(feature_map, range(num_qubits))
    
    return qc
