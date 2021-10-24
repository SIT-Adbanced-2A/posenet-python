# Throw Detecter
This program detects the throwing of an object.  

# Requirements
- python3.6
- tensorflow-gpu
- scipy
- pyyaml
- cython

# Getting Started
1. Build library
```
cd mylib
setup.py build_ext --inplace
```

2. Execute Throw Detecter
```
cd ..
python throw_detecter.py movie_file
```