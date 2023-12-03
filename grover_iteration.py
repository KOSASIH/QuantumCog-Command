def grover_iteration(circuit, n, marked_items):
    # Apply the inversion about the average
    circuit.h(range(n))
    circuit.x(range(n))
    circuit.h(n-1)
    circuit.mct(list(range(n-1)), n-1)
    circuit.h(n-1)
    circuit.x(range(n))
    circuit.h(range(n))
    
    # Apply the oracle
    apply_oracle(circuit, n, marked_items)
