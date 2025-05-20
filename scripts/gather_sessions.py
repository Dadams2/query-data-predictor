import os
import pandas as pd
import re

# this script extists in case dataset generation fails
def gather_session_files(directory):
    # Regex to match the file pattern and extract the session number
    pattern = r"query_prediction_session_(\d+)\.pkl"
    data = []

    # Walk through the directory to find matching files
    for root, _, files in os.walk(directory):
        for file in files:
            match = re.match(pattern, file)
            if match:
                session_id = int(match.group(1))
                absolute_path = os.path.abspath(os.path.join(root, file))
                data.append({"session_id": session_id, "path": absolute_path})

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data, columns=["session_id", "path"])
    return df

# example usage
if __name__ == "__main__":
    directory = "../data/datasets" 
    df = gather_session_files(directory)
    print(df)
    # Write the DataFrame to a CSV file named "metadata.csv" in the same directory
    output_path = os.path.join(directory, "metadata.csv")
    df.to_csv(output_path, index=False)
    print(f"Metadata written to {output_path}")
