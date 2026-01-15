import sys
import os
import numpy as np
import pandas as pd
import scipy.io

def average_field_roi(scouts_file, mesh_file, e_normal_file, region_names_file, output_file):
    """
    Calculates the average E-normal field for each ROI  defined in the atlas.

    Args:
        scouts_file (str): Path to the .mat file containing ROIs.
        mesh_file (str): Path to the .vtk mesh file (cortex).
        e_normal_file (str): Path to the text file containing E-normal scalars.
        region_names_file (str): Path to .txt file with abbreviated region names.
        output_file (str): Path to save the resulting .txt table.
    """

    print(f"Processing E-Normals from: {os.path.basename(e_normal_file)}")

    # 1. Load E-Normal Scalars
    # ------------------------
    df = pd.read_csv(e_normal_file, sep=r'\s+', header=None, engine='python')

    # Extract coordinates (Cols 4,5,6; 12,13,14; 20,21,22) and scalars (Cols 7, 15, 23)
    coordinates = np.concatenate([
        df.iloc[:, [4, 5, 6]].values,
        df.iloc[:, [12, 13, 14]].values,
        df.iloc[:, [20, 21, 22]].values
    ], axis=0)

    scalars = np.concatenate([
        df.iloc[:, 7].values,
        df.iloc[:, 15].values,
        df.iloc[:, 23].values
    ], axis=0)

    # Create DataFrame and deduplicate
    new_df = pd.DataFrame({'Coordinates': list(map(tuple, coordinates)), 'Scalar': scalars})
    df_e_field = new_df.drop_duplicates()
    df_e_field['Coordinates'] = df_e_field['Coordinates'].apply(lambda x: tuple(round(coord, 3) for coord in x))

    print(f"Loaded {len(df_e_field)} unique E-field data points.")

    # 2. Load Mesh Vertices (.vtk)
    # ---------------------------
    print(f"Loading Mesh: {os.path.basename(mesh_file)}")
    vertices = []

    with open(mesh_file, 'r') as file:
        # Skip header
        for _ in range(5):
            next(file)

        for line in file:
            if line.strip().startswith("POLYGONS") or line.strip().startswith("CELLS"):
                break
            # Parse vertex coordinates
            try:
                parts = list(map(float, line.strip().split()))
                if len(parts) == 3:
                    vertices.append(parts)
            except ValueError:
                continue

    vertices = np.array(vertices)
    print(f"Loaded {len(vertices)} mesh vertices.")

    # Convert to pandas dataframe for merging
    df_vertices = pd.DataFrame(vertices, columns=['X', 'Y', 'Z'])
    df_vertices['Coordinates'] = df_vertices.apply(lambda row: (row['X'], row['Y'], row['Z']), axis=1)
    df_vertices = df_vertices.drop(columns=['X', 'Y', 'Z'])
    df_vertices['Coordinates'] = df_vertices['Coordinates'].apply(lambda x: tuple(round(coord, 3) for coord in x))

    # Initialize Scalar column with NaN or 0
    df_vertices['Scalar'] = np.nan

    # 3. Map E-Field to Mesh Vertices
    # -------------------------------
    print("Mapping E-field data to Mesh...")
    # Merge on Coordinates
    merged_df = df_vertices.merge(df_e_field, on='Coordinates', how='left', suffixes=('', '_y'))

    # Update Scalar values
    df_vertices['Scalar'] = merged_df['Scalar_y']

    # Check for missing attributions
    missing_count = df_vertices['Scalar'].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} mesh vertices could not be matched to E-field data.")

    # 4. Load Abbreviated Region Names
    # --------------------------------
    abbr_names = []
    if os.path.exists(region_names_file):
        print(f"Loading abbreviated names from: {os.path.basename(region_names_file)}")
        with open(region_names_file, 'r', encoding='utf-8') as f:
            abbr_names = [line.strip() for line in f if line.strip()]
    else:
        print(f"Warning: Region names file not found: {region_names_file}")

    # 5. Process ROIs
    # ------------------------
    print(f"Processing Scouts from: {os.path.basename(scouts_file)}")
    mat = scipy.io.loadmat(scouts_file)

    scouts_data = mat['Scouts']

    roi_results = []

    # Check if it's a 1D array of structs
    if scouts_data.ndim == 2:
        scouts_list = scouts_data[0]
    else:
        scouts_list = scouts_data

    # Validate length match if we are replacing names
    if abbr_names and len(abbr_names) != len(scouts_list):
        print(f"Warning: Number of scouts ({len(scouts_list)}) matches number of names ({len(abbr_names)})? No.")
        print("Will attempt to map by index.") #mismatch will occur for thalamus it's normal. We don't have E_normal for thalamus.

    for i, scout in enumerate(scouts_list):
        # Extract Label (Default to internal label)
        label = scout['Label'][0]

        # Override with abbreviated name if available
        if i < len(abbr_names):
            label = abbr_names[i]

        # Extract Vertices
        verts = scout['Vertices']

        # Handle different shapes of Vertices array
        if verts.shape[0] == 1:
            verts = verts[0]
        else:
            verts = verts.flatten()

        # Convert 1-based matlab indices to 0-based python  indices
        verts_idx = [int(v) - 1 for v in verts]

        # 6. Compute Average for this Scout
        # -------------------------------
        # Select scalar values for these vertices
        roi_scalars = df_vertices.loc[verts_idx, 'Scalar']

        # Mean ignoring NaNs
        avg_val = roi_scalars.mean()

        roi_results.append({
            'Region': label,
            'Average_Enormal': avg_val
        })

    # 7. Save Output
    # --------------
    df_results = pd.DataFrame(roi_results)

    # Save as csv
    print(f"Saving results to: {output_file}")
    df_results.to_csv(output_file, sep='\t', index=False, float_format='%.6f')

    print("\nResult Preview:")
    print(df_results.head())

if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Inputs:
    SCOUTS_MAT = os.path.join(base_dir, 'data', 'region_names_and_scouts', 'scout_whole_brain.mat')
    MESH_VTK = os.path.join(base_dir, 'data', 'cortex_mesh', 'cortex_mesh_template.vtk')
    ENORMAL_TXT = os.path.join(base_dir, 'data', 'E_normals/E_normal_BIP_NE2LOC1')
    REGION_NAMES_TXT = os.path.join(base_dir, 'data', 'region_names_and_scouts', 'region_names_whole_brain.txt')

    OUTPUT_TXT = os.path.join(base_dir, 'data', 'roi_averaged_enormals_BIP_NE2LOC1.txt')

    #running script:
    
    average_field_roi(SCOUTS_MAT, MESH_VTK, ENORMAL_TXT, REGION_NAMES_TXT, OUTPUT_TXT)
