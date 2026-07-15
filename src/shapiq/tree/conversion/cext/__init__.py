# Fake cext module to bypass import errors
def create_edge_tree_arrays(*args, **kwargs):
    raise NotImplementedError("C++ extension not available")
