from PIL import Image
import os
import numpy as np

# Define the input and output directories
input_dir = "/data2/qinghua/ZDJ/datasets_111/Mapillary/validation/labelTrainIds/"
output_dir = "/data2/qinghua/ZDJ/datasets_111/Mapillary/validation/labelTrainIds_color19/"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define Cityscapes color map for 19 classes + void (index 255)
color_map = [
    (128, 64, 128),   # 0: road
    (244, 35, 232),   # 1: sidewalk
    (70, 70, 70),     # 2: building
    (102, 102, 156),  # 3: wall
    (190, 153, 153),  # 4: fence
    (153, 153, 153),  # 5: pole
    (250, 170, 30),   # 6: traffic light
    (220, 220, 0),    # 7: traffic sign
    (107, 142, 35),   # 8: vegetation
    (152, 251, 152),  # 9: terrain
    (70, 130, 180),   # 10: sky
    (220, 20, 60),    # 11: person
    (255, 0, 0),      # 12: rider
    (0, 0, 142),      # 13: car
    (0, 0, 70),       # 14: truck
    (0, 60, 100),     # 15: bus
    (0, 80, 100),     # 16: train
    (0, 0, 230),      # 17: motorcycle
    (119, 11, 32),    # 18: bicycle
    (0, 0, 0)         # 255: void (for non-matching Mapillary classes)
]

# Function to convert Mapillary trainIds to Cityscapes colored label map
def create_cityscapes_colored_label_map(input_path, output_path, color_map):
    # Open the input image (Mapillary trainIds mask)
    mask = Image.open(input_path).convert('L')  # Convert to grayscale if not already
    mask_array = np.array(mask)

    # Map Mapillary trainIds to Cityscapes trainIds (0-18, others to 255)
    # Simplified mapping: 0-18 map directly, 19-254 to 255
    cityscapes_mask = np.where((mask_array >= 0) & (mask_array <= 18), mask_array, 255)

    # Create a new RGB image with the same size
    colored_mask = np.zeros((cityscapes_mask.shape[0], cityscapes_mask.shape[1], 3), dtype=np.uint8)

    # Assign Cityscapes colors based on mapped indices
    for class_idx in range(len(color_map)):
        colored_mask[cityscapes_mask == class_idx] = color_map[class_idx]

    # Save the colored image
    colored_image = Image.fromarray(colored_mask)
    colored_image.save(output_path)

# Process all images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('_labelTrainIds.png'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('_labelTrainIds.png', '_cityscapes_color.png'))
        create_cityscapes_colored_label_map(input_path, output_path, color_map)

print(f"Colored label maps with Cityscapes colors have been saved to {output_dir}")