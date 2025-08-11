def get_forbidden_pairs(affinity_matrix, categories):
    forbidden = set()
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            a, b = categories[i], categories[j]
            if affinity_matrix.loc[a, b] == 0.0:
                forbidden.add((a, b))
    return forbidden
