from qiskit.circuit.library import EfficientSU2

def construct_variational_circuit(num_qubits, num_layers):
    qc = QuantumCircuit(num_qubits)
    
    variational_form = EfficientSU2(num_qubits, reps=num_layers)
    qc.append(variational_form, range(num_qubits))
    
    return qc
