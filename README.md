# fakeonion
Fake Onion Headlines 

Strongly recommend using the `fakeonion.yml`, but if you want to install manually:
```bash
conda create -n fakeonion pytorch=1.9 cudatoolkit=11.1 streamlit=0.89.0 fuzzywuzzy=0.18.0 transformers=4.11 -y
conda activate fakeonion
# stopping here gave me issues when running streamlit: AttributeError: module 'google.protobuf.descriptor' has no attribute '_internal_create_key'.
python -m pip install --upgrade protobuf
```
After initial install of 
