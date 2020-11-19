import os
import sys
from PIL import Image

input_file = sys.argv[1]
image_dir = sys.argv[2]
output_dir = sys.argv[3]

if not os.path.exists(output_dir):
    os.system('mkdir -p %s/images' % output_dir)
    os.system('mkdir -p %s/labels' % output_dir)

state = 0  # 0: filename 1: bbx num 2: bbx
filename = ''
bbx_num = 0
current_bbx_index = 0
file_handler = None
image_height = 0
image_width = 0
class_type = 0
classes = set()
with open(input_file, 'r') as label_file:
    while True:
        lines = label_file.readlines(100000)
        if not lines:
            break
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if state == 0:
                image_file_path = os.path.join(image_dir, line)
                # os.system('cp %s %s' % (image_file_path, os.path.join(output_dir, 'images', line.split('/')[-1])))
                image = Image.open(image_file_path)
                image_height = image.height
                image_width = image.width
                del image
                filename = line.split('/')[-1].split('.')[0] + '.txt'
                # class_type = int(line.split('--')[0])
                # if class_type == 61:
                #     class_type = 60
                class_type = 0
                classes.add(line.split('--')[-1].split('/')[0])
                file_handler = open(os.path.join(output_dir, 'labels', filename), 'w')
                state = 1
            elif state == 1:
                bbx_num = int(line)
                state = 2
            elif state == 2:
                data = line.split()
                x1 = int(data[0])
                y1 = int(data[1])
                w = int(data[2])
                h = int(data[3])
                if w > 0 and h > 0:
                    x1 /= image_width
                    y1 /= image_height
                    w /= image_width
                    h /= image_height

                    x1 = max(min(x1, 1), 0)
                    y1 = max(min(y1, 1), 0)
                    w = max(min(w, 1), 0)
                    h = max(min(h, 1), 0)

                    file_handler.write('%d %f %f %f %f\n' % 
                    (class_type, x1 + w / 2, y1 + h / 2, w, h))
                current_bbx_index += 1
                if current_bbx_index == bbx_num:
                    file_handler.close()
                    file_handler = None
                    state = 0
                    current_bbx_index = 0
            index += 1
print(list(classes))


                
            