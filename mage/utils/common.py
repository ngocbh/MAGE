def get_hash_key(S, T):
    sorted_S = sorted(S)
    sorted_T = sorted(T)
    return hash(tuple(sorted_S + ["#"] + sorted_T))
