def apply_oracle(circuit, n, marked_items):
    # Apply a phase flip to the marked items
    for item in marked_items:
        circuit.z(item)
