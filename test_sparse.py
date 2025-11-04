from dyno.sparse_sym import SymTensor

def test_to_array():
    """Test the SymTensor toarray method with a 3D tensor."""
    # We try with a 3d tensor
    data = {
        (0, 0, 0): 1,
        (0, 1, 2): 4,
        (1, 1, 1): 2,
        (2, 2, 2): 3,
    }
    shape = (3, 3, 3)
    t = SymTensor(data, shape)
    arr = t.toarray()
    
    # Check shape
    assert arr.shape == shape
    
    # Check that symmetry is preserved (dimensions 1,2 should be symmetric)
    assert arr[0,1,2] == arr[0,2,1] == 4
    assert arr[0,0,0] == 1
    assert arr[1,1,1] == 2
    assert arr[2,2,2] == 3
    
    print("SymTensor test passed!")

if __name__ == "__main__":
    test_to_array()
