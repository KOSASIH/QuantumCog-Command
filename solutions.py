database = ['item1', 'item2', 'item3', 'item4']
marked_items = [2]  # Index of the target item(s) in the database

solutions = grover_search(database, marked_items)
print("Solutions:", solutions)
