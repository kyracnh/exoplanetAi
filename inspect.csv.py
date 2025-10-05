import csv

def inspect_csv(filepath='./Data/real_data.csv'):
    """Inspect CSV file structure to diagnose parsing issues."""
    
    print("=" * 60)
    print("CSV FILE INSPECTION")
    print("=" * 60)
    
    # Read first 100 lines to get past comments
    with open(filepath, 'r', encoding='utf-8') as f:
        all_lines = [f.readline() for _ in range(100)]
    
    # Find where comments end
    header_line_idx = None
    for i, line in enumerate(all_lines):
        if not line.strip().startswith('#') and line.strip():
            header_line_idx = i
            break
    
    print(f"\nComment lines found: {header_line_idx}")
    print("\nLast 5 comment lines:")
    print("-" * 60)
    if header_line_idx:
        for i in range(max(0, header_line_idx - 5), header_line_idx):
            print(f"Line {i+1}: {all_lines[i][:100].strip()}")
    
    print("\nFirst data line (header):")
    print("-" * 60)
    if header_line_idx is not None:
        print(f"Line {header_line_idx+1}: {repr(all_lines[header_line_idx][:200])}")
        lines = all_lines[header_line_idx:header_line_idx+5]
    else:
        lines = all_lines[:10]
        print("Could not find header - showing first 10 lines")
        for i, line in enumerate(lines, 1):
            print(f"Line {i}: {repr(line[:200])}")
    
    # Detect delimiter
    print("\n" + "=" * 60)
    print("DELIMITER DETECTION")
    print("=" * 60)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        sample = f.read(4096)
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            print(f"Detected delimiter: {repr(dialect.delimiter)}")
            print(f"Quote character: {repr(dialect.quotechar)}")
        except:
            print("Could not auto-detect delimiter")
    
    # Count common delimiters in first data line
    first_line = lines[0] if lines else ""
    print(f"\nDelimiter counts in first data line:")
    print(f"  Commas (,): {first_line.count(',')}")
    print(f"  Tabs (\\t): {first_line.count(chr(9))}")
    print(f"  Pipes (|): {first_line.count('|')}")
    print(f"  Semicolons (;): {first_line.count(';')}")
    print(f"  Spaces: {first_line.count(' ')}")
    
    # Try to read with different delimiters, skipping comment lines
    print("\n" + "=" * 60)
    print("COLUMN DETECTION ATTEMPTS")
    print("=" * 60)
    
    for delim_name, delim in [('comma', ','), ('tab', '\t'), ('pipe', '|'), ('semicolon', ';')]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # Skip comment lines
                for line in f:
                    if not line.strip().startswith('#'):
                        # This is the header
                        header = line.strip().split(delim)
                        break
                
                print(f"\nWith {delim_name} delimiter ({repr(delim)}):")
                print(f"  Number of columns: {len(header)}")
                if len(header) <= 20:
                    print(f"  Column names: {header}")
                else:
                    print(f"  First 10 columns: {header[:10]}")
                    print(f"  Last 5 columns: {header[-5:]}")
                
                # Check if our required columns are present
                required = ['koi_disposition', 'koi_score', 'koi_depth', 'koi_model_snr']
                found = [col for col in required if col in header]
                if found:
                    print(f"  ✓ Found required columns: {found}")
                    
        except Exception as e:
            print(f"\nWith {delim_name} delimiter: ERROR - {e}")

if __name__ == "__main__":
    inspect_csv()