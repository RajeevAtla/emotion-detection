import os
import sys
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tf_events(log_dir, output_csv):
    print(f"[Extractor] Reading events from: {log_dir}")
    
    # Initialize EventAccumulator to load all data (scalars)
    # size_guidance=0 loads all events without downsampling
    event_acc = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    event_acc.Reload()

    # Get all scalar tags (e.g., 'Loss/train', 'Accuracy/val')
    tags = event_acc.Tags()['scalars']
    
    if not tags:
        print("[Extractor] No scalar tags found in TensorBoard logs.")
        return

    print(f"[Extractor] Found tags: {tags}")

    # Dictionary to hold dataframes for merging
    data_frames = []

    for tag in tags:
        events = event_acc.Scalars(tag)
        
        # Extract steps and values
        steps = [e.step for e in events]
        values = [e.value for e in events]
        wall_times = [e.wall_time for e in events]
        
        # Create a temp dataframe for this tag
        df = pd.DataFrame({
            'step': steps,
            tag: values,
            # 'wall_time': wall_times # Optional: uncomment if you need timestamps
        })
        
        # Set step as index for easy merging
        df.set_index('step', inplace=True)
        data_frames.append(df)

    # Merge all dataframes on 'step' (outer join to handle different logging frequencies)
    if data_frames:
        final_df = pd.concat(data_frames, axis=1)
        final_df.sort_index(inplace=True)
        final_df.reset_index(inplace=True)
        
        print(f"[Extractor] Saving merged data to {output_csv}")
        print(final_df.head())
        final_df.to_csv(output_csv, index=False)
        print("[Extractor] Success.")
    else:
        print("[Extractor] Failed to aggregate data.")

def main():
    tb_dir = os.environ.get("TENSORBOARD_DIR")
    out_csv = os.environ.get("OUTPUT_CSV")

    if not os.path.exists(tb_dir):
        print(f"ERROR: TensorBoard directory not found: {tb_dir}")
        sys.exit(1)

    extract_tf_events(tb_dir, out_csv)

if __name__ == "__main__":
    main()
