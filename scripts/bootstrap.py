# On your local machine or in S3, create a bootstrap script
cat > bootstrap_python_libs.sh << 'EOF'
#!/bin/bash
set -x

# Install Python libraries on ALL nodes (master and core)
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud

# Also install for Python 2 (if needed for compatibility)
sudo pip install numpy pandas matplotlib --upgrade || true

# Verify installations
python3 -c "import numpy; print('NumPy version:', numpy.__version__)"
python3 -c "import pandas; print('Pandas version:', pandas.__version__)"
python3 -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"
EOF