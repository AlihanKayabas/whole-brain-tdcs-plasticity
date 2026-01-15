#convert off to vtk: 
def convert_off_to_vtk(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    # Extract the total number of points (vertices) from the second line
    header = lines[1].strip().split()
    num_points = int(header[0])
    num_faces = int(header[1])

    # Prepare the VTK file header
    vtk_header = [
        "# vtk DataFile Version 3.0",
        "vtk output",
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {num_points} float"
    ]

    # Extract the points (vertices) and faces (polygons) from the OFF file
    points = lines[2:2 + num_points]
    faces = lines[2 + num_points:2 + num_points + num_faces]

    # Prepare the polygons line
    polygons_line = f"POLYGONS {num_faces} {num_faces * 6}"

    # Write to the VTK file
    with open(output_file, 'w') as outfile:
        # Write the header
        outfile.write("\n".join(vtk_header) + "\n")

        # Write the points
        outfile.writelines(points)

        # Write the polygons header
        outfile.write(polygons_line + "\n")

        # Write the faces
        for face in faces:
            values = face.strip().split()
            n_vertices = int(values[0])  # Typically 3 for triangles
            vertices = " ".join(values[1:])
            outfile.write(f"{n_vertices} {vertices}\n")

convert_off_to_vtk('D:/patient2/MRI/pipeline_patient2/cortex_mesh_patient2_15000V.off', 'D:/patient2/MRI/pipeline_patient2/cortex_mesh_patient2_15000V.vtk')
