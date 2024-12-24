import os
import subprocess
import random
import math
import json
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from numpy import mean
import pandas as pd

def run_proteinmpnn(file_name, fixed_positions, chain_id):
    """
    Run ProteinMPNN for a given file and chain with fixed positions.

    Args:
        file_name (str): Name of the JSONL file without extension.
        fixed_positions (list): List of fixed positions for the chain.
        chain_id (str): Chain ID to be used.

    Returns:
        tuple: Designed sequences and path to the FASTA file.
    """
    designed_sequences = []
    jsonl_path = f"{file_name}.jsonl"
    output_folder = f"{file_name}_designs"
    fasta_path = f"{output_folder}/seqs/{file_name}.fa"

    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} does not exist.")
        return designed_sequences, fasta_path

    make_fixed_positions_command = [
        "python",
        "../../ProteinMPNN/helper_scripts/make_fixed_positions_dict.py",
        "--specify_non_fixed",
        "--position_list",
        ' '.join(map(str, fixed_positions)),
        "--chain_list",
        chain_id,
        "--input_path",
        jsonl_path,
        "--output_path",
        f"{file_name}_fixed_pos.jsonl"
    ]

    try:
        subprocess.run(make_fixed_positions_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running make_fixed_positions_dict.py for {file_name}: {e}")
        return designed_sequences, fasta_path

    protein_mpnn_command = [
        "python",
        "../../ProteinMPNN/protein_mpnn_run.py",
        "--jsonl_path",
        jsonl_path,
        "--fixed_positions_jsonl",
        f"{file_name}_fixed_pos.jsonl",
        "--out_folder",
        output_folder,
        "--num_seq_per_target",
        "1",
        "--pdb_path_chains",
        chain_id,
        "--sampling_temp",
        "0.1",
        "--seed",
        "0",
        "--batch_size",
        "1",
        "--model_name",
        "v_48_020"
    ]

    try:
        subprocess.run(protein_mpnn_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running protein_mpnn_run.py for {file_name}: {e}")
        return designed_sequences, fasta_path

    try:
        if not os.path.exists(fasta_path):
            print(f"Error: {fasta_path} does not exist.")
            return designed_sequences, fasta_path

        with open(fasta_path, "r") as fasta_file:
            records = list(SeqIO.parse(fasta_file, "fasta"))
            if len(records) >= 2:
                designed_amino_acid_sequence = str(records[1].seq)
                designed_sequences.append({
                    f"seq_chain_{chain_id}": designed_amino_acid_sequence,
                })
            else:
                print(f"Error: {file_name} does not contain multiple sequences.")
    except FileNotFoundError as e:
        print(f"Error reading {fasta_path}: {e}")

    return designed_sequences, fasta_path

def calculate_solubility(fasta_path, output_path):
    """
    Calculate the solubility of a given FASTA file using NetSolP tool.

    Args:
        fasta_path (str): Path to the input FASTA file.
        output_path (str): Path to save the solubility results.

    Returns:
        float: solubility value.
    """
    command = f"python predict.py --FASTA_PATH {fasta_path} --OUTPUT_PATH {output_path} --MODEL_TYPE ESM12 --PREDICTION_TYPE S"
    try:
        subprocess.run(command, shell=True, check=True)
        df = pd.read_csv(output_path)
        return df.iloc[1, 7]  # Adjust column index if necessary
    except subprocess.CalledProcessError as e:
        print("Error calculating solubility:", e)
        return 0.0

def calculate_instability(sequence):
    """
    Calculate the instability and flexibility of a protein sequence.

    Args:
        sequence (str): Protein sequence.

    Returns:
        tuple: Instability index and average flexibility.
    """
    analyzer = ProteinAnalysis(sequence)
    return analyzer.instability_index(), mean(analyzer.flexibility())

def metropolis_optimization(iterations, file_name, fasta_file_path, chain_id):
    """
    Perform Metropolis optimization to find the best protein sequence.

    Args:
        iterations (int): Number of iterations for the optimization process.
        file_name (str): Name of the JSONL file without extension.
        fasta_file_path (str): Path to the input FASTA file.
        chain_id (str): Chain ID to be optimized.

    Returns:
        tuple: Best sequence and its solubility value.
    """
    jsonl_path = f"{file_name}.jsonl"
    with open(jsonl_path, 'r') as json_file:
        json_data = json.load(json_file)

    initial_sequence = json_data[f"seq_chain_{chain_id}"]
    current_solubility = calculate_solubility(fasta_file_path, "output_solubility.csv")
    current_instability, current_flexibility = calculate_instability(initial_sequence)

    best_sequence = initial_sequence
    best_solubility = current_solubility

    log_file_path = f"{file_name}_optimization_results.log"
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"File name: {file_name}\n")
        log_file.write(f"Initial sequence: {initial_sequence}\n")
        log_file.write(f"Initial instability: {current_instability}\n")
        log_file.write(f"Initial flexibility: {current_flexibility}\n")
        log_file.write(f"Initial solubility: {current_solubility}\n\n")

    for _ in range(iterations):
        random_position = random.randint(0, len(initial_sequence) - 1)
        new_sequences, new_fasta_path = run_proteinmpnn(file_name, [random_position], chain_id)
        if not new_sequences:
            print(f"Error: No new sequences generated for {file_name}.")
            continue

        new_sequence = new_sequences[0][f"seq_chain_{chain_id}"]
        new_solubility = calculate_solubility(new_fasta_path, "output_solubility.csv")
        new_instability, new_flexibility = calculate_instability(new_sequence)

        if (new_solubility > current_solubility and
            new_instability < 40 and
            new_instability < current_instability and
            new_flexibility > current_flexibility):

            json_data[f"seq_chain_{chain_id}"] = new_sequence
            current_solubility = new_solubility
            current_instability, current_flexibility = new_instability, new_flexibility

            if new_solubility > best_solubility:
                best_sequence = new_sequence
                best_solubility = new_solubility

            with open(jsonl_path, 'w') as updated_json:
                json.dump(json_data, updated_json)

    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Best sequence: {best_sequence}\n")
        log_file.write(f"Best solubility: {best_solubility}\n")
        log_file.write(f"Final flexibility: {current_flexibility}\n")
        log_file.write(f"Final instability: {current_instability}\n\n")

    return best_sequence, best_solubility

# Example usage
if __name__ == "__main__":
    iterations = 3000
    cath_file_path = 'cath_name.txt'
    with open(cath_file_path, 'r') as cath_file:
        file_names = [line.strip() for line in cath_file.readlines()]

    for file_name in file_names:
        fasta_file_path = f"{file_name}_designs/seqs/{file_name}.fa"
        chain_id = file_name[4].upper()
        metropolis_optimization(iterations, file_name, fasta_file_path, chain_id)
