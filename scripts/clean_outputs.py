import argparse
import os
import glob
import sys
import shutil

def get_run_files(run_dir):
    return glob.glob(os.path.join(run_dir, "*.jsonl"))

def parse_run_id_timestamp(filename):
    # format: ..._YYYYMMDD-HHMMSS_hash.jsonl
    # We want to sort by timestamp.
    # We can just rely on file mtime if filename parsing is brittle, 
    # but filename is explicit.
    try:
        base = os.path.basename(filename)
        # Extract YYYYMMDD-HHMMSS
        # It's usually the second to last part if split by "_"
        parts = base.replace(".jsonl", "").split("_")
        # Find parts that look like timestamp
        for p in parts:
            if "-" in p and len(p) == 15 and p[0].isdigit():
                return p
        return "00000000-000000"
    except:
        return "00000000-000000"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="outputs/runs")
    parser.add_argument("--keep_latest", action="store_true", help="Keep only the latest run files")
    parser.add_argument("--all", action="store_true", help="Delete plots and summaries too")
    args = parser.parse_args()
    
    files = get_run_files(args.run_dir)
    if not files:
        print("No files to clean.")
        return
        
    if args.keep_latest:
        # Group by run_id (timestamp + hash)
        # We can just sort all files by modification time or name, 
        # but a "run" spans multiple files.
        # We need to identify the LATEST run_id.
        
        # Collect timestamps
        timestamps = set()
        file_map = {} # timestamp -> [files]
        
        for f in files:
            ts = parse_run_id_timestamp(f)
            timestamps.add(ts)
            if ts not in file_map: file_map[ts] = []
            file_map[ts].append(f)
            
        sorted_ts = sorted(list(timestamps))
        if not sorted_ts:
            print("Could not parse file timestamps. Aborting to be safe.")
            return
            
        latest = sorted_ts[-1]
        print(f"Latest timestamp detected: {latest}")
        
        # Delete everything else
        for ts in sorted_ts[:-1]:
            print(f"Deleting run group: {ts} ({len(file_map[ts])} files)")
            for f in file_map[ts]:
                os.remove(f)
                
    else:
        # Delete ALL runs?
        print(f"Deleting {len(files)} files in {args.run_dir}...")
        for f in files:
            os.remove(f)
            
    if args.all:
        # Clean summaries and plots
        print("Cleaning summaries and plots...")
        for f in glob.glob("outputs/summaries/*"): os.remove(f)
        for f in glob.glob("outputs/plots/*"): os.remove(f)
        
    print("Done.")

if __name__ == "__main__":
    main()
