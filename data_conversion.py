import os
import json
import csv
import numpy as np
import skimage.io as io

def pipistrel_to_coco(original_annotation_file, outfile=None, with_other_objects=False, with_nature=False):

    # load annotations
    with open(original_annotation_file, 'r') as f_in:
        csv_f = csv.reader(f_in)

        original_annotations = []
        for row in csv_f:
            original_annotations.append(row)

    # set up output
    annotations = {'images': [], 'categories': [], 'annotations': []}

    # define categories
    annotations['categories'].append({'id': 1, 'name': 'boat'})
    if with_other_objects:
        annotations['categories'].append({'id': 2, 'name': 'bowwave'})
        annotations['categories'].append({'id': 3, 'name': 'undefObj'})
    if with_nature:
        annotations['categories'].append({'id': 4, 'name': 'nature'})

    
    # create image annotations
    filenames = list(np.unique([original_annotations[i][0] for i in range(1,len(original_annotations))]))
    for i, filename in enumerate(filenames):
#         if i % 10 == 0:
#             print("{}/{}".format(i, len(filenames)), end="\r")

        # Load image to extract height and width
    #     file_path = os.path.join(image_dir, filename)
    #     img_data = io.imread(file_path)

        img_ann = {
            'file_name': filename,
            'height': 2464,
            'width': 3280,
            'id': i
        }
        annotations['images'].append(img_ann)

    # create instance annotations
    j = 0
    for i, orig_ann in enumerate(original_annotations):
        if i == 0:
            continue

        if orig_ann[1] == 'boat':
            category_id = 1
        elif orig_ann[1] == 'bowwave' and with_other_objects:
            category_id = 2
        elif orig_ann[1] == 'undefObj' and with_other_objects:
            category_id = 3
        elif orig_ann[1] == 'nature' and with_nature:
            category_id = 4
        else:
            continue

        x, y = [int(orig_ann[2]), int(orig_ann[3])]
        dx = int(orig_ann[4]) - int(orig_ann[2])
        dy = int(orig_ann[5]) - int(orig_ann[3])
        filename = orig_ann[0]
        img_id = [
            annotations['images'][k]['id'] 
            for k in range(len(annotations['images'])) 
            if annotations['images'][k]['file_name'] == filename
        ][0]

        ann = {
            'area': dx * dy,
            'bbox': [x, y, dx, dy],
            'category_id': category_id,
            'id': j,
            'image_id': img_id,
            'iscrowd': 0,
        }
        annotations['annotations'].append(ann)
        j += 1
    
    if outfile is not None:
        with open(outfile, 'w') as f_out:
            json.dump(annotations, f_out)
        
    return annotations