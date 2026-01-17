#!/bin/bash
set -x

echo "BOOTSTRAP STARTED on $(hostname)"

# Ensure pip exists
sudo yum install -y python3-pip

# Upgrade pip safely
sudo python3 -m pip install --upgrade pip

# Install libraries
sudo python3 -m pip install \
numpy pandas matplotlib seaborn scikit-learn nltk pyarrow --no-cache-dir

# Test imports
python3 - <<EOF
import numpy
import pandas
import matplotlib
print("ALL LIBS INSTALLED OK")
EOF

echo "BOOTSTRAP COMPLETED"
