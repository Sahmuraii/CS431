import pandas as pd

def convert_txt_to_excel(input_file: str, output_file: str):
    try:
        # Define the column names explicitly
        columns = [
            "video ID", "uploader", "age", "category", "length", 
            "views", "rate", "ratings", "comments", "related IDs"
        ]
        
        # Read the tab-separated text file without assuming headers
        df = pd.read_csv(input_file, sep='\\t', header=None)
        
        # Extract the first 9 columns as named and combine the rest as "related IDs"
        df.columns = [*columns[:-1], *[f"related_{i}" for i in range(1, df.shape[1] - 9 + 1)]]
        df["related IDs"] = df.iloc[:, 9:].apply(lambda x: ",".join(x.dropna()), axis=1)
        
        # Keep only the required columns
        selected_df = df[columns]
        
        # Write to Excel
        selected_df.to_excel(output_file, index=False, engine='openpyxl')

        print(f"Data successfully written to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
if __name__ == "__main__":
    input_txt = "0.txt"  # Replace with your input file path
    output_excel = "output_file.xlsx"  # Replace with your desired output file path
    convert_txt_to_excel(input_txt, output_excel)
