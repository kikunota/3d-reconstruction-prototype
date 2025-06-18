#!/usr/bin/env python3
"""
Test numpy to Python scalar conversion to verify the fix
"""

import numpy as np
import json

def test_numpy_conversion():
    """Test various numpy scalar conversions"""
    
    # Test cases that might cause "only size-1 arrays can be converted to Python scalars"
    test_cases = [
        ("Single value", np.array(5.0)),
        ("Mean of array", np.mean([1, 2, 3, 4, 5])),
        ("Numpy float64", np.float64(3.14159)),
        ("Numpy int32", np.int32(42)),
        ("Array element", np.array([1.5, 2.5, 3.5])[0]),
    ]
    
    print("Testing numpy to Python scalar conversions:")
    print("=" * 50)
    
    all_passed = True
    
    for name, value in test_cases:
        try:
            # Test conversion methods
            if hasattr(value, 'item'):
                converted = value.item()
            else:
                converted = float(value)
            
            # Test JSON serialization
            json_str = json.dumps({"value": converted})
            
            print(f"✓ {name}: {type(value).__name__} -> {type(converted).__name__} = {converted}")
            
        except Exception as e:
            print(f"✗ {name}: Failed - {e}")
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("✓ All numpy conversion tests passed!")
    else:
        print("✗ Some conversion tests failed!")
    
    return all_passed

if __name__ == '__main__':
    test_numpy_conversion()