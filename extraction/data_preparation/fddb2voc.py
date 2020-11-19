import numpy as np
import sys
import os
from PIL import Image

train_file_list = [
    'FDDB-fold-01-ellipseList.txt',
    'FDDB-fold-02-ellipseList.txt',
    'FDDB-fold-03-ellipseList.txt',
    'FDDB-fold-04-ellipseList.txt',
    'FDDB-fold-05-ellipseList.txt',
    'FDDB-fold-06-ellipseList.txt',
    'FDDB-fold-07-ellipseList.txt',
    'FDDB-fold-08-ellipseList.txt',
    'FDDB-fold-09-ellipseList.txt'
]

test_file_list = [
    'FDDB-fold-10-ellipseList.txt'
]

root_dir = sys.argv[1]

def get_ellipse_param(major_radius, minor_radius, angle):
    a, b = major_radius, minor_radius
    sin_theta = np.sin(-angle)
    cos_theta = np.cos(-angle)
    A = a**2 * sin_theta**2 + b**2 * cos_theta**2
    B = 2 * (a**2 - b**2) * sin_theta * cos_theta
    C = a**2 * cos_theta**2 + b**2 * sin_theta**2
    F = -a**2 * b**2
    return A, B, C, F


def calculate_rectangle(A, B, C, F):
    y = np.sqrt(4*A*F / (B**2 - 4*A*C))
    y1, y2 = -np.abs(y), np.abs(y)
    
    x = np.sqrt(4*C*F / (B**2 - 4*C*A))
    x1, x2 = -np.abs(x), np.abs(x)
    
    return (x1, y1), (x2, y2)

def get_rectangle(major_radius, minor_radius, angle, center_x, center_y):
    A, B, C, F = get_ellipse_param(major_radius, minor_radius, angle)
    p1, p2 = calculate_rectangle(A, B, C, F)
    return (center_x+p1[0], center_y+p1[1]), (center_x+p2[0], center_y+p2[1])

output_root = sys.argv[2]
if not os.path.exists(output_root):
    os.system('mkdir -p %s/train/images' % output_root)
    os.system('mkdir -p %s/train/labels' % output_root)
    os.system('mkdir -p %s/test/images' % output_root)
    os.system('mkdir -p %s/test/labels' % output_root)

for split in ['train', 'test']:
    current_file_list = train_file_list if split == 'train' else test_file_list
    for fold_file in current_file_list:
        fold_file_path = os.path.join(root_dir, fold_file)
        fold_file = open(fold_file_path, 'r')
        bbx_num = 0
        filename = ''
        current_bbx_index = 0
        file_handler = None
        image_height = 0
        image_width = 0
        state = 0
        while True:
            lines = fold_file.readlines(100000)
            if not lines:
                break
            index = 0
            while index < len(lines):
                line = lines[index].strip()
                if state == 0:
                    image_file_path = line + '.jpg'
                    image_name = line.split('/')[-1]
                    os.system('cp %s %s' % (image_file_path, os.path.join(output_root, split, 'images')))
                    image = Image.open(image_file_path)
                    image_height = image.height
                    image_width = image.width
                    del image
                    filename = image_name.split('.')[0] + '.txt'
                    file_handler = open(os.path.join(output_root, split, 'labels', filename), 'w')
                    state = 1
                elif state == 1:
                    bbx_num = int(line)
                    state = 2
                elif state == 2:
                    data = line.split()
                    major_radius = float(data[0])
                    minor_radius = float(data[1])
                    angle = float(data[2])
                    center_x = float(data[3])
                    center_y = float(data[4])
                    (x1, y1), (x2, y2) = get_rectangle(major_radius, minor_radius, angle, center_x, center_y)
                    x1 /= image_width
                    y1 /= image_height
                    x2 /= image_width
                    y2 /= image_height
                    
                    x1 = max(min(x1, 1), 0)
                    y1 = max(min(y1, 1), 0)
                    x2 = max(min(x2, 1), 0)
                    y2 = max(min(y2, 1), 0)
                    w = x2 - x1
                    h = y2 - y1
                    if w > 0 and h > 0:
                        file_handler.write('0 %f %f %f %f\n' % ((x1 + x2) / 2, (y1 + y2) / 2, w, h))
                    current_bbx_index += 1
                    if current_bbx_index == bbx_num:
                        file_handler.close()
                        file_handler = None
                        state = 0
                        current_bbx_index = 0
                index += 1
        fold_file.close()
