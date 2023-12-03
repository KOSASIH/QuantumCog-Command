from qiskit import QuantumCircuit, Aer, execute

def initialize_state(circuit, n):
    for i in range(n):
        circuit.h(i)
