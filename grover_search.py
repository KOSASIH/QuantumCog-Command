def grover_search(database, marked_items):
    n = len(database)
    num_iterations = int(round(np.sqrt(n)))
    
    # Create a quantum circuit with n qubits
    circuit = QuantumCircuit(n, n)
    
    # Step 1: Initialize the quantum state
    initialize_state(circuit, n)
    
    # Step 2 and 3: Apply the oracle and perform the Grover iteration
    for _ in range(num_iterations):
        grover_iteration(circuit, n, marked_items)
    
    # Measure the quantum state
    circuit.measure(range(n), range(n))
    
    # Simulate the circuit
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circuit, simulator, shots=1)
    result = job.result().get_counts(circuit)
    
    # Extract the solution(s) from the measurement result
    solutions = [int(key[::-1], 2) for key in result.keys()]
    
    return solutions
