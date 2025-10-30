import pandas as pd
import ast

def preprocess_and_flatten(csv_path, save_path=None, num_frames=30, num_objects=10):
    """
    Full preprocessing pipeline for object-sequence CSVs:
      1. Replace '0' entries with '[0,0,0,0]'
      2. Add numeric suffixes (_1, _2, ...) to duplicate video IDs
      3. Flatten sequential frame data into per-video chunks

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.
    save_path : str, optional
        Path to save the final flattened CSV. If None, overwrites the input file.
    num_frames : int, optional (default=30)
        Number of frames per sequence chunk.
    num_objects : int, optional (default=10)
        Number of object columns per frame.

    Returns
    -------
    pd.DataFrame
        Flattened DataFrame ready for model input.
    """

    # === STEP 1: Load CSV ===
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    print("✅ Loaded CSV:", csv_path)
    print(f"Shape before processing: {df.shape}")

    # === STEP 2: Replace '0' with '[0,0,0,0]' in object columns ===
    cols_to_update = df.columns[2:2+num_objects]  # assuming first 2 are video_id, frame
    df[cols_to_update] = df[cols_to_update].replace('0', '[0,0,0,0]')
    print("✅ Replaced '0' with '[0,0,0,0]' in object columns")

    # === STEP 3: Add suffixes for duplicate video IDs ===
    df['video_id'] = df.groupby('video_id').cumcount().add(1).astype(str).radd(df['video_id'] + '_')
    print("✅ Added numeric suffixes for duplicate video IDs")

    # === STEP 4: Flatten video sequences ===
    columns = ['video_id', 'frame']
    for i in range(num_frames):
        for j in range(num_objects):
            columns.append(f'{i}_obj_{j}')
    columns.append('target')

    flattened_df = pd.DataFrame(columns=columns)

    for vid in df['video_id'].unique():
        temp_df = df[df['video_id'] == vid].reset_index(drop=True)

        # Fill missing rows to make count multiple of num_frames
        fill_tuple = (vid, 0, *["[0,0,0,0]"]*num_objects, 0)
        current_rows = len(temp_df)
        rows_to_add = (num_frames - (current_rows % num_frames)) % num_frames

        if rows_to_add > 0:
            new_rows_df = pd.DataFrame([fill_tuple]*rows_to_add, columns=temp_df.columns)
            temp_df = pd.concat([temp_df, new_rows_df], ignore_index=True)

        print(f"{vid}: Final rows = {len(temp_df)} (multiple of {num_frames}: {len(temp_df) % num_frames == 0})")

        # Flatten every chunk of `num_frames`
        for start in range(0, len(temp_df), num_frames):
            chunk = temp_df.iloc[start:start+num_frames]
            if len(chunk) < num_frames:
                continue

            row = {'video_id': vid}
            row['frame'] = chunk['frame'].iloc[0] if 'frame' in chunk.columns else chunk['frame_no'].iloc[0]

            for i, (_, frame_row) in enumerate(chunk.iterrows()):
                obj_cols = frame_row.index[2:-1]  # skip video_id & frame, skip target
                for j, col in enumerate(obj_cols):
                    val = frame_row[col]
                    if isinstance(val, str) and val.startswith('['):
                        val = ast.literal_eval(val)
                    row[f'{i}_obj_{j}'] = val

            row['target'] = chunk['target'].max()
            flattened_df = pd.concat([flattened_df, pd.DataFrame([row], columns=columns)], ignore_index=True)

    # === STEP 5: Save final output ===
    if save_path is None:
        save_path = csv_path.replace('.csv', '_flattened.csv')
    flattened_df.to_csv(save_path, index=False)

    print("\n✅ Final flattened DataFrame saved to:", save_path)
    print(flattened_df.head())
    print(f"Shape: {flattened_df.shape}")

    return flattened_df
