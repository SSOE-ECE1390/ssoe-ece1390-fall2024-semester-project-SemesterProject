import subprocess
import sys

def install_package(package_name):
    try:
        # Check if the package is already installed
        __import__(package_name)
        print(f"'{package_name}' is already installed.")
    except ImportError:
        # If not installed, install the package
        print(f"'{package_name}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Install numpy
install_package("numpy")