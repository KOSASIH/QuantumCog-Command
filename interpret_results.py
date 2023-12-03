# Interpreting the Results for TSP

# Convert the classical bits into a sequence of cities
def interpret_results(results):
    sequence = []
    for key in results.keys():
        sequence.append([int(bit) for bit in key])
    return sequence

# Calculate the total distance of the route
def calculate_distance(sequence):
    total_distance = 0
    for i in range(len(sequence) - 1):
        total_distance += distance(sequence[i], sequence[i + 1])
    return total_distance

# Obtain the optimal solution
def get_optimal_solution(results):
    sequence = interpret_results(results)
    optimal_sequence = min(sequence, key=calculate_distance)
    return optimal_sequence

# Example usage
results = {'0000': 0.2, '0001': 0.1, '0010': 0.3, '0011': 0.4}
optimal_sequence = get_optimal_solution(results)
print("Optimal Sequence:", optimal_sequence)
