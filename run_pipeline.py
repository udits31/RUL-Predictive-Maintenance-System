"""
Complete pipeline runner - executes all steps in sequence.
"""
import subprocess
import sys
import os

def run_step(step_name, command):
    """Run a pipeline step and handle errors."""
    print("\n" + "="*70)
    print(f"STEP: {step_name}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, command],
            check=True,
            capture_output=False
        )
        print(f"✓ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {step_name} failed with error code {e.returncode}")
        return False

def main():
    """Run complete pipeline."""
    print("\n" + "="*70)
    print("NASA CMAPSS PREDICTIVE MAINTENANCE - COMPLETE PIPELINE")
    print("="*70)
    
    steps = [
        ("Data Generation", "generate_data.py"),
        ("Preprocessing", "preprocessing.py"),
        ("Model Training", "train_models.py")
    ]
    
    for step_name, script in steps:
        if not run_step(step_name, script):
            print(f"\n✗ Pipeline failed at: {step_name}")
            sys.exit(1)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Start the API server: python app.py")
    print("2. Open frontend/index.html in your browser")
    print("\nAll models and data are ready in the models/ directory.")

if __name__ == '__main__':
    main()
