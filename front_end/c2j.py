import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    # List to store the data
    data = []
    
    # Read the CSV file
    with open(csv_file_path, 'r') as csv_file:
        # Create CSV reader with header
        csv_reader = csv.DictReader(csv_file)
        
        # Convert each row to a dictionary and append to data list
        for row in csv_reader:
            # Convert string values to appropriate types (float for numbers)
            processed_row = {
                'lon_bin': float(row['lon_bin']),
                'lat_bin': float(row['lat_bin']),
                'yesterday_pred': int(row['yesterday_pred']),
                'yesterday_true': float(row['yesterday_true']),
                'today_pred': int(row['today_pred']),
                'yesterday_diff': float(row['yesterday_diff'])
            }
            data.append(processed_row)
    
    # Write to JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    # Replace these with your actual file paths
    input_csv = "./data/deployed_files/visualization_data.csv"
    output_json = "./front_end/o1.json"
    
    try:
        csv_to_json(input_csv, output_json)
        print(f"Successfully converted {input_csv} to {output_json}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")