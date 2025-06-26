import h5py
import numpy as np
from transformers import AutoTokenizer

# Load Qwen tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load Lotka-Volterra dataset
def load_data(file_path="lotka_volterra_data.h5"):
    with h5py.File(file_path, "r") as f:
        trajectories = f["trajectories"][:]
        time_points = f["time"][:]           
    return trajectories, time_points

# LLMTIME Preprocessing Function
def preprocess_time_series(data, alpha=10, decimal_places=2):
    """
    Converts numerical time series into a text-based format compatible with Qwen's tokenizer.
    
    Parameters:
      data: input the Lotka-Volterra dataset
      alpha: scales the data by alpha
      decimal_places: Rounds to 'decimal_places'
      
    Returns:
      Formats time series using commas (,) for variables and semicolons (;) for timesteps.
    """
    # Scale and round values
    scaled_data = np.round(data / alpha, decimals=decimal_places)

    # Convert to formatted text sequence
    sequence_strings = []
    for system in scaled_data:  # Loop over each system
        formatted_timesteps = [",".join(map(str, timestep)) for timestep in system]
        sequence_strings.append(";".join(formatted_timesteps))  # Join timesteps

    return sequence_strings

# Tokenize sequence using Qwen
def tokenize_sequence(sequence):
    """
    Tokenizes the preprocessed sequence using Qwen2.5 tokenizer.
    """
    tokenized = tokenizer(sequence, return_tensors="pt")["input_ids"]
    return tokenized.tolist()[0]  # Convert tensor to list

# Example usage
if __name__ == "__main__":
    trajectories, time_points = load_data()

    # Preprocess first system
    example_sequences = preprocess_time_series(trajectories[:2])  # First two systems

    # Show preprocessed and tokenized results
    for i, seq in enumerate(example_sequences):
        print(f"Example {i+1}:")
        print("Preprocessed Sequence:", seq[:100], "...")  # Truncate for readability
        print("Tokenized Sequence:", tokenize_sequence(seq)[:20], "...")  # Show first 20 tokens
        print("-" * 50)
        
        
        
        
def decode_sequence(encoded_sequence, alpha=10):
    """
    Converts a tokenized text sequence back into a numerical array.
    Parameters:
      encoded_sequence: sequence to decode 
      alpha: scaling factor
      
     Splits by semicolon for timesteps.
     Splits by comma for prey & predator values.
    
    Converts back to floats and rescales using alpha.
    """
    time_series = []
    for timestep in encoded_sequence.split(";"):  # Split into timesteps
        # Ensure timestep has both prey and predator values
        values = timestep.split(",")
        if len(values) != 2:  
            print(f"Skipping invalid timestep: '{timestep}'")  # Debugging output
            continue  # Skip invalid entries
        
        try:
            prey, predator = map(float, values)  # Convert to float
            time_series.append([prey * alpha, predator * alpha])  # Rescale
        except ValueError:
            print(f"Error parsing values: {values}")  # Debugging output

    return np.array(time_series) if time_series else np.array([[0, 0]])
