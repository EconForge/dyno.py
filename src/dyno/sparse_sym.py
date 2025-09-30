class SymTensor:
    """A simple symmetric tensor class.

    A tensor that is symmetric in all dimensions except the first one.
    For example, a tensor T[i,j,k,l] where T[i,j,k,l] = T[i,k,j,l] = T[i,l,j,k] etc.

    Attributes:
        data: The data of the tensor, stored in coo format (a dict).
        shape: The shape of the tensor.
        syms: List of symmetry groups (optional, defaults to all dims except first being symmetric).
    """

    def __init__(self, data, shape, syms=None):
        self.data = data
        self.shape = shape
        if syms is None:
            # Default: all dimensions except the first are symmetric with each other
            if len(shape) > 1:
                self.syms = [list(range(1, len(shape)))]
            else:
                self.syms = []
        else:
            self.syms = syms

    def toarray(self):
        import numpy as np
        from itertools import permutations

        arr = np.zeros(self.shape)

        for indices, value in self.data.items():
            # Add the original entry
            arr[indices] = value

            # Generate all symmetric permutations
            for sym_group in self.syms:
                if len(sym_group) <= 1:
                    continue

                # Get the indices for this symmetry group
                sym_indices = [indices[i] for i in sym_group]
                # Get all unique permutations of these indices
                for perm in permutations(sym_indices):
                    # Create new index tuple with permuted values
                    new_indices = list(indices)
                    for i, new_val in zip(sym_group, perm):
                        new_indices[i] = new_val
                    new_indices = tuple(new_indices)

                    # Check bounds and set value
                    if all(
                        0 <= new_indices[i] < self.shape[i]
                        for i in range(len(self.shape))
                    ):
                        arr[new_indices] = value

        return arr
