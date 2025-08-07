#!/usr/bin/env python3
"""
Simple test runner for SVM implementation tests.
"""
import subprocess
import sys

def run_tests():
    """Run the SVM unit tests."""
    print("Running SVM Implementation Unit Tests")
    print("=" * 50)
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "test_svm_classifier.py", 
            "-v", 
            "--tb=short"
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("🎉 ALL TESTS PASSED!")
            print("Your SVM implementation is working correctly.")
        else:
            print("\n" + "=" * 50)
            print("❌ Some tests failed.")
            print("Please check the output above for details.")
            
        return result.returncode
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
