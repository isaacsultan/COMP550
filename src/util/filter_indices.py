import pickle

filtered_indices_path = "dumps/filtered_indices.pkl"
url_tag = "__url__"
path_tag = "__path__"

def get_filtered_indices():
    with open(filtered_indices_path, "rb") as f:
        return pickle.load(f)

def _internal_find_index(data, i, filtered_indices):
    for j in range(10):
        k = i*10 + j
        row = data[k]
        for l in range(0, 2):
            if url_tag in row[l] or path_tag in row[l]:
                # Add 0, 9 indices in filtered indices
                filtered_indices.extend([n for n in range(i*10, i*10 + 10)])
                return


def generate_filtered_indices(data):
    # Data now contains indices, so we get the indices for __url__ and __path__
    # and find these in the data
    filtered_indices = []

    batch_range = range(len(data)//10) # Should be divisible by 10

    for i in batch_range:
        _internal_find_index(data, i, filtered_indices)

    return filtered_indices


if __name__ == "__main__":
    pass
    # Get v2 data
    print("Loading validation file...")
    with open("dumps/v2/valid_expanded.pkl", "rb") as f:
        valid_set = pickle.load(f)
    print("Done.")

    # Get indices where there is urls and paths
    print("Filtering indices...")
    indices = generate_filtered_indices(valid_set)

    print("Done.")

    with open(filtered_indices_path, "wb") as f:
        pickle.dump(indices, f)