import pandas as pd
import argparse
import os
import json

def process_arxiv_data(input_file, output_file, domain=None, max_papers=None):
    """
    Process the arXiv dataset to create a filtered version focused on a specific domain.
    
    Args:
        input_file (str): Path to the input arXiv file (JSON or CSV)
        output_file (str): Path to write the processed file (JSON or CSV)
        domain (str, optional): Domain filter (e.g., 'cs', 'physics', 'math')
        max_papers (int, optional): Maximum number of papers to include
    """
    print(f"Loading arXiv data from {input_file}...") 
    
    # Determine file type based on extension
    file_extension = os.path.splitext(input_file)[1].lower()
    
    if file_extension == '.json':
        # For JSON files - use pandas read_json with lines=True for JSONL format
        # This is more memory efficient for large files
        try:
            # First try regular JSON format
            df = pd.read_json(input_file)
            print("Loaded data using standard JSON format")
        except ValueError:
            # If that fails, try JSONL format (JSON Lines - one JSON object per line)
            print("Standard JSON loading failed. Trying JSON Lines format...")
            try:
                # Process in chunks to handle large files
                chunks = []
                total_papers = 0
                for chunk in pd.read_json(input_file, lines=True, chunksize=10000):
                    # Clean the data
                    if 'title' in chunk.columns and 'abstract' in chunk.columns:
                        chunk = chunk.dropna(subset=['title', 'abstract'])
                    
                    # Apply domain filter if specified
                    if domain and 'categories' in chunk.columns:
                        chunk = chunk[chunk['categories'].str.contains(domain, case=False, na=False)]
                    
                    chunks.append(chunk)
                    total_papers += len(chunk)
                    
                    print(f"Processed chunk... Total papers so far: {total_papers}")
                    
                    # Break if we've reached the maximum number of papers
                    if max_papers and total_papers >= max_papers:
                        break
                
                if not chunks:
                    raise ValueError("No valid data found after processing")
                    
                # Combine all chunks
                df = pd.concat(chunks, ignore_index=True)
                print(f"Successfully loaded {len(df)} records from JSONL format")
            except Exception as e:
                print(f"Error loading JSON Lines format: {e}")
                print("Falling back to manual line-by-line processing (slower but more robust)...")
                
                # Manual line-by-line processing as last resort
                data = []
                processed_lines = 0
                valid_lines = 0
                
                with open(input_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                            
                        try:
                            record = json.loads(line)
                            
                            # Apply filters early to save memory
                            if 'title' not in record or 'abstract' not in record:
                                continue
                            if not record.get('title') or not record.get('abstract'):
                                continue
                            if domain and 'categories' in record:
                                if domain.lower() not in record['categories'].lower():
                                    continue
                                    
                            data.append(record)
                            valid_lines += 1
                            
                            if valid_lines % 1000 == 0:
                                print(f"Processed {line_num} lines, found {valid_lines} valid records")
                                
                            if max_papers and valid_lines >= max_papers:
                                break
                                
                        except json.JSONDecodeError:
                            if processed_lines < 5:  # Only show first few errors
                                print(f"Warning: Invalid JSON at line {line_num}, skipping")
                            processed_lines += 1
                            continue
                
                if not data:
                    raise ValueError("Could not parse any valid JSON objects from the file")
                    
                # Convert to DataFrame
                df = pd.DataFrame(data)
                print(f"Successfully loaded {len(df)} records using manual processing")
    elif file_extension == '.csv':
        # For CSV files - read in chunks to handle large files
        chunked_data = pd.read_csv(input_file, chunksize=10000)
        
        # Process each chunk
        chunks = []
        total_papers = 0
        
        for chunk in chunked_data:
            # Clean the data
            chunk = chunk.dropna(subset=['title', 'abstract'])
            
            # Apply domain filter if specified
            if domain:
                chunk = chunk[chunk['categories'].str.contains(domain, case=False, na=False)]
            
            chunks.append(chunk)
            total_papers += len(chunk)
            
            print(f"Processed chunk... Total papers so far: {total_papers}")
            
            # Break if we've reached the maximum number of papers
            if max_papers and total_papers >= max_papers:
                break
        
        # Combine all chunks
        df = pd.concat(chunks, ignore_index=True)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Please use .json or .csv files.")
    
    # Additional filtering and cleaning - this applies for both JSON and CSV after initial loading
    # Clean the data if not already done in chunk processing
    if 'title' in df.columns and 'abstract' in df.columns:
        df = df.dropna(subset=['title', 'abstract'])
    
    # Apply domain filter if specified and not already applied in chunk processing
    if domain and 'categories' in df.columns:
        before_count = len(df)
        df = df[df['categories'].str.contains(domain, case=False, na=False)]
        print(f"Filtered for domain '{domain}': kept {len(df)} out of {before_count} papers")
    
    # Limit to max_papers if specified
    if max_papers and len(df) > max_papers:
        df = df.iloc[:max_papers]
        print(f"Limited to {max_papers} papers as requested")
    
    # Select relevant columns
    columns_to_keep = ['title', 'abstract', 'categories']
    if 'authors' in df.columns:
        columns_to_keep.append('authors')
    
    # Only keep columns that exist in the dataframe
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    df = df[existing_columns]
    
    # Determine output format based on extension
    output_extension = os.path.splitext(output_file)[1].lower()
    
    if output_extension == '.json':
        # Save to JSON
        print(f"Saving {len(df)} papers to {output_file} in JSON format...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
    elif output_extension == '.csv':
        # Save to CSV
        print(f"Saving {len(df)} papers to {output_file} in CSV format...")
        df.to_csv(output_file, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_extension}. Please use .json or .csv for output files.")
    
    print(f"Successfully saved {len(df)} papers to {output_file}")
    
    # Print domain distribution
    if 'categories' in df.columns:
        print("\nTop 10 categories in the processed dataset:")
        
        # First detect the format of the categories column
        try:
            # Check the first non-null value
            sample_categories = df['categories'].dropna().iloc[0] if not df['categories'].dropna().empty else None
            
            if sample_categories:
                if isinstance(sample_categories, str):
                    # If it's a string, try to split it
                    try:
                        # Try space-separated format (most common)
                        category_counts = df['categories'].str.split().explode().value_counts().head(10)
                    except:
                        # If that fails, try comma-separated format
                        try:
                            category_counts = df['categories'].str.split(',').explode().str.strip().value_counts().head(10)
                        except:
                            # Last resort: use the whole string as a category
                            category_counts = df['categories'].value_counts().head(10)
                elif isinstance(sample_categories, list):
                    # If it's already a list, explode it directly
                    category_counts = df['categories'].explode().value_counts().head(10)
                else:
                    # If it's some other format, just use as is
                    category_counts = df['categories'].value_counts().head(10)
                
                # Print the results
                for category, count in category_counts.items():
                    print(f"{category}: {count}")
            else:
                print("No category data found in the filtered dataset")
        except Exception as e:
            # Handle any other exceptions
            print(f"Could not analyze categories distribution: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arXiv dataset for the domain expert chatbot")
    parser.add_argument("--input", required=True, help="Path to the input arXiv file (JSON or CSV)")
    parser.add_argument("--output", required=True, help="Path to write the processed file (JSON or CSV)")
    parser.add_argument("--domain", help="Domain filter (e.g., 'cs', 'physics', 'math')")
    parser.add_argument("--max_papers", type=int, help="Maximum number of papers to include")
    
    args = parser.parse_args()
    
    process_arxiv_data(args.input, args.output, args.domain, args.max_papers)