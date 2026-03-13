# import os
# import shutil

# # Base directory containing all the folders
# base_dir = r"D:\Machine learning\Arch technologhy\YOLO11\orignal_dataset_polygon_to_bounding_box\segmented_results\New folder"

# # Loop through each folder in the base directory
# for folder_name in os.listdir(base_dir):
#     folder_path = os.path.join(base_dir, folder_name)

#     if os.path.isdir(folder_path):
#         files = os.listdir(folder_path)

#         for file in files:
#             file_path = os.path.join(folder_path, file)
#             new_path = os.path.join(base_dir, file)

#             # Move the file to base directory
#             shutil.move(file_path, new_path)
#             print(f"Moved '{file}' to '{base_dir}'")

#         # After moving, delete the now-empty folder
#         if not os.listdir(folder_path):  # Check if folder is empty
#             os.rmdir(folder_path)
#             print(f"Deleted empty folder '{folder_name}'")


import os
import shutil

# Base directory containing all the folders
base_dir = r"D:\Machine learning\Arch technologhy\YOLO11\orignal_dataset_polygon_to_bounding_box\segmented_results\New folder"

# Loop through each folder in the base directory
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)

    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)

        for i, file in enumerate(files):
            file_path = os.path.join(folder_path, file)
            # Create a new filename using the folder name and a unique index
            new_filename = (
                f"{folder_name}_{i}.jpg"  # Use index to avoid name collisions
            )
            new_path = os.path.join(base_dir, new_filename)

            # Move the file to base directory with the new name
            shutil.move(file_path, new_path)
            print(f"Moved '{file}' to '{new_path}'")

        # After moving, delete the now-empty folder
        if not os.listdir(folder_path):  # Check if folder is empty
            os.rmdir(folder_path)
            print(f"Deleted empty folder '{folder_name}'")
