# HMHO
Heuristic Metropolis-Hastings Optimization Algorithm (HMHO)

## Overview
This project provides a Python-based implementation for optimizing protein sequences using Metropolis optimization and external tools like ProteinMPNN and NetSolP. The code allows for the design and analysis of protein sequences, improving their physical properties such as stability and resolution.

## Features
- **ProteinMPNN Integration**: Generate protein sequences with specific fixed positions.
- **NetSolP Integration**: Calculate resolution of protein sequences using a deep learning model.
- **Metropolis Optimization**: Iteratively refine sequences to improve biophysical properties.
- **Sequence Analysis**: Evaluate sequence instability and flexibility.

## Requirements
The project relies on the following dependencies:

### Python Packages
Install the required Python packages using:
```bash
pip install -r requirements.txt
```
- `biopython`
- `numpy`
- `pandas`

### External Tools
#### ProteinMPNN
1. Clone the repository:
   ```bash
   git clone https://github.com/dauparas/ProteinMPNN.git
   ```
2. Navigate to the directory:
   ```bash
   cd ProteinMPNN
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Verify the setup:
   ```bash
   python helper_scripts/make_fixed_positions_dict.py --help
   ```

#### NetSolP
1. Clone the repository:
   ```bash
   git clone https://github.com/TviNet/NetSolP-1.0.git
   ```
2. Navigate to the directory:
   ```bash
   cd NetSolP-1.0
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Test the setup:
   ```bash
   python predict.py --help
   ```

## Usage
### Input Files
1. **JSONL Files**: Input protein sequence data in JSONL format (e.g., `example.jsonl`).
   ```json
   {"seq_chain_A": "MKRESHKHAEQARRNRLAVALHELASLIPAEWKQQNVSAAPSKATTVEAACRYIRHLQQNGST", "coords_chain_A": {"N_chain_A": [[3.43, -2.059, 57.593], [4.503, -1.273, 55.106], [6.764, -1.617, 53.515], [7.337, -2.021, 50.761], [5.017, -1.711, 49.22], [4.582, 0.937, 49.934], [5.266, 2.436, 47.935], [3.87, 4.624, 46.058], [4.594, 6.744, 47.609], [6.897, 7.564, 46.61], [6.785, 8.744, 43.715], [4.307, 10.33, 44.043], [5.393, 12.123, 45.641], [7.642, 12.655, 44.296], [6.776, 14.022, 42.089], [6.85, 16.764, 42.645], [9.648, 17.311, 42.557], [9.755, 16.909, 39.871], [8.327, 19.099, 38.508], [9.756, 21.05, 39.794], [11.978, 20.909, 38.489], [10.652, 21.715, 36.332], [9.54, 24.523, 36.903], [11.816, 25.563, 36.952], [12.166, 25.521, 34.493], [10.31, 27.34, 33.528], [11.164, 30.041, 34.193], [13.46, 30.237, 32.72], [15.321, 31.184, 29.937], [14.531, 30.564, 26.559], [15.669, 31.934, 24.502], [18.169, 32.237, 25.581], [19.444, 29.701, 25.817], [20.556, 29.18, 23.115], [23.168, 29.865, 23.357], [24.036, 27.974, 25.114], [25.378, 24.724, 24.478], [23.663, 22.355, 22.743], [22.267, 19.861, 24.766], [20.257, 17.991, 25.022], [16.723, 17.849, 25.129], [16.038, 17.345, 28.21], [14.917, 16.05, 31.516], [16.916, 16.592, 33.329], [19.222, 17.178, 31.955], [18.724, 19.942, 31.674], [18.776, 21.005, 34.161], [21.162, 21.433, 34.186], [21.369, 23.694, 32.648], [19.812, 25.819, 34.154], [21.653, 26.638, 35.907], [23.623, 27.828, 34.413], [22.243, 30.231, 33.411], [21.418, 31.434, 35.898], [24.007, 31.44, 37.334], [24.952, 33.312, 35.62], [22.883, 35.722, 35.358], [23.408, 36.895, 37.865], [26.048, 38.607, 37.989], [25.059, 41.093, 36.534], [25.504, 43.588, 34.201], [27.939, 44.523, 35.202], [31.087, 45.887, 34.586]], "CA_chain_A": [[4.785, -2.49, 57.148], [4.43, -0.953, 53.669], [8.138, -1.716, 53.025], [7.329, -2.451, 49.377], [3.839, -1.042, 48.69], [4.622, 2.363, 50.262], [5.438, 2.853, 46.544], [3.362, 5.95, 45.702], [5.331, 7.662, 48.456], [7.945, 8.055, 45.736], [6.296, 9.705, 42.724], [3.357, 11.305, 44.593], [6.304, 13.036, 46.299], [8.597, 12.893, 43.192], [6.091, 14.974, 41.222], [7.375, 18.074, 42.932], [10.944, 17.403, 41.932], [9.527, 16.957, 38.433], [8.003, 20.499, 38.186], [10.825, 21.869, 40.347], [12.772, 20.93, 37.294], [9.795, 22.516, 35.441], [9.476, 25.977, 37.124], [13.199, 25.805, 36.66], [12.029, 25.646, 33.05], [9.492, 28.527, 33.224], [11.937, 31.271, 34.275], [14.724, 30.039, 32.031], [15.133, 31.749, 28.594], [14.848, 29.797, 25.379], [16.602, 32.855, 23.822], [19.478, 32.061, 26.199], [19.787, 28.373, 25.317], [21.384, 29.174, 21.877], [24.514, 30.018, 23.894], [24.356, 26.625, 25.548], [25.587, 23.579, 23.586], [22.494, 21.546, 23.01], [22.533, 18.583, 25.437], [19.044, 17.262, 24.628], [15.665, 18.385, 26.004], [15.771, 16.378, 29.294], [14.569, 16.308, 32.922], [18.19, 16.412, 33.979], [20.084, 17.982, 31.091], [18.411, 21.361, 31.794], [19.126, 21.435, 35.494], [22.474, 21.916, 33.8], [21.081, 25.032, 32.19], [19.394, 26.805, 35.179], [22.872, 27.022, 36.561], [24.139, 28.768, 33.41], [21.42, 31.433, 33.489], [21.611, 31.872, 37.274], [25.419, 31.798, 37.479], [25.009, 34.582, 34.867], [22.017, 36.787, 35.85], [24.082, 37.507, 39.018], [26.84, 39.775, 37.574], [24.412, 41.791, 35.416], [25.928, 44.958, 33.947], [29.31, 44.668, 35.709], [31.966, 46.118, 33.448]], "C_chain_A": [[4.821, -2.46, 55.629], [5.793, -0.89, 52.96], [8.324, -2.273, 51.616], [6.228, -1.644, 48.672], [3.768, 0.46, 48.984], [4.751, 3.145, 48.94], [5.163, 4.307, 46.178], [4.072, 7.086, 46.434], [6.443, 8.292, 47.628], [7.447, 9.139, 44.804], [5.305, 10.744, 43.265], [4.132, 12.414, 45.335], [7.3, 13.547, 45.235], [8.094, 13.988, 42.255], [6.629, 16.385, 41.397], [8.704, 18.185, 42.22], [10.824, 17.491, 40.409], [9.363, 18.427, 37.973], [9.06, 21.456, 38.737], [11.754, 22.018, 39.17], [11.987, 21.813, 36.308], [9.681, 24.04, 35.66], [10.881, 26.508, 36.866], [13.227, 25.974, 35.154], [11.215, 26.888, 32.658], [10.252, 29.876, 33.238], [13.34, 31.114, 33.706], [14.364, 30.54, 30.639], [15.516, 30.908, 27.381], [15.92, 30.62, 24.639], [18.033, 32.781, 24.37], [20.065, 30.837, 25.518], [20.684, 28.245, 24.075], [22.881, 29.246, 22.208], [24.897, 28.616, 24.33], [24.327, 25.548, 24.441], [24.355, 22.72, 23.801], [22.871, 20.178, 23.624], [21.522, 17.575, 24.95], [17.987, 17.825, 25.564], [15.266, 17.39, 27.113], [15.596, 16.887, 30.721], [15.787, 16.105, 33.815], [19.168, 17.329, 33.269], [19.896, 19.52, 31.196], [18.664, 21.906, 33.197], [20.507, 21.998, 35.203], [22.345, 23.394, 33.5], [20.751, 26.095, 33.244], [20.557, 27.376, 35.96], [23.427, 28.135, 35.696], [23.567, 30.174, 33.527], [21.733, 32.135, 34.802], [23.032, 32.337, 37.506], [25.578, 33.143, 36.796], [24.217, 35.781, 35.45], [22.596, 37.563, 37.035], [24.872, 38.769, 38.605], [26.245, 40.508, 36.368], [24.832, 43.267, 35.307], [27.357, 45.322, 34.297], [30.267, 44.848, 34.543], [33.197, 46.98, 33.716]], "O_chain_A": [[5.138, -3.464, 54.967], [5.94, -0.201, 51.946], [9.3, -2.977, 51.346], [6.496, -0.898, 47.738], [3.01, 1.186, 48.316], [4.481, 4.338, 48.88], [6.092, 5.083, 45.958], [4.116, 8.233, 45.962], [6.871, 9.413, 47.898], [7.621, 10.317, 45.099], [5.511, 11.938, 43.065], [3.6, 13.482, 45.616], [7.661, 14.739, 45.221], [8.865, 14.829, 41.788], [6.889, 17.085, 40.43], [8.829, 18.975, 41.283], [11.653, 18.128, 39.735], [10.194, 18.936, 37.191], [9.201, 22.558, 38.243], [12.234, 23.094, 38.865], [12.583, 22.604, 35.572], [9.693, 24.785, 34.673], [11.089, 27.654, 36.439], [14.196, 26.463, 34.609], [11.45, 27.466, 31.594], [10.013, 30.748, 32.392], [14.295, 31.728, 34.181], [13.23, 30.34, 30.188], [16.697, 30.597, 27.172], [17.01, 30.097, 24.342], [18.984, 33.245, 23.724], [21.027, 30.928, 24.766], [21.484, 27.3, 23.999], [23.731, 28.774, 21.441], [25.913, 28.083, 23.887], [23.459, 25.547, 23.532], [24.007, 22.399, 24.943], [23.776, 19.485, 23.142], [21.88, 16.469, 24.57], [18.302, 18.253, 26.684], [14.229, 16.727, 26.998], [16.138, 17.931, 31.102], [15.675, 15.711, 34.964], [19.94, 18.044, 33.892], [20.786, 20.3, 30.81], [18.718, 23.11, 33.416], [20.951, 22.953, 35.845], [22.996, 24.225, 34.135], [21.354, 27.163, 33.226], [20.431, 28.373, 36.669], [23.598, 29.248, 36.164], [24.304, 31.154, 33.674], [22.282, 33.239, 34.803], [23.231, 33.486, 37.9], [26.17, 34.038, 37.38], [24.826, 36.792, 35.865], [22.322, 38.754, 37.166], [24.404, 39.889, 38.848], [26.877, 40.583, 35.302], [24.624, 44.068, 36.237], [27.894, 46.31, 33.764], [30.316, 44.01, 33.633], [33.561, 47.145, 34.904]]}, "name": "1a0aA00", "num_of_chains": 1, "seq": "MKRESHKHAEQARRNRLAVALHELASLIPAEWKQQNVSAAPSKATTVEAACRYIRHLQQNGST"}

   ```
2. **CATH Name File**: Text file listing JSONL filenames (e.g., `cath_name.txt`).
   ```
   1a0aA00
   1a00B00
   1a0cA00
   1a0gA02
   1a0hA01
   ```

### Running the Code
1. Update the paths for ProteinMPNN and NetSolP tools in the script if necessary.
2. Execute the main script:
   ```bash
   python protein_optimization.py
   ```

### Output
- **Optimized Sequences**: Stored in updated JSONL files.
- **Logs**: Detailed logs of the optimization process.
- **FASTA Files**: Generated sequences in FASTA format.

## Code Explanation
### Main Functions
1. **run_proteinmpnn**:
   - Uses ProteinMPNN to generate sequences with fixed positions.
2. **calculate_resolution**:
   - Computes resolution using NetSolP.
3. **calculate_instability**:
   - Analyzes instability and flexibility of sequences.
4. **metropolis_optimization**:
   - Optimizes sequences using a Metropolis algorithm.

### Example
Run the optimization pipeline with default settings:
```python
python protein_optimization.py
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- [NetSolP](https://github.com/TviNet/NetSolP-1.0)
- [BioPython](https://biopython.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)


