import os
import numpy as np
import glob
from utils.xml_parser import XMLParser
from utils import rej_size, rej_table, replace_table


def generate_labels_info(out_path="./data/labels_info.npy"):
    home_dir_path = "/home/Dataset/Papsmear/original/"
    file_types = ["SS/*/*.jpg", "SS2/*/*.jpg"]

    img_path_list = []
    for ftype in file_types:
        img_path_list.extend(glob.glob(home_dir_path + ftype))
    
    print("Num of Images: {}".format(len(img_path_list)))

    labels_info = {}
    for idx, img_path in enumerate(img_path_list):
        xml_path = img_path[:-3] + "xml"
        if os.path.isfile(xml_path):
            parser = XMLParser(xml_path)
            if (parser.height, parser.width) not in rej_size:
                ID = img_path.split("original/")[-1]            
                labels = parser.objects
                new_labels = []
                for label in labels:
                    cname, xmin, ymin, xmax, ymax = label
                    if cname in rej_table:
                        continue
                    if xmin >= xmax or ymin >= ymax:
                        continue                        

                    if cname in replace_table:
                        cname = replace_table[cname]
                    new_labels.append([xmin, ymin, xmax, ymax, cname])
                
                # Add new refined labels
                labels_info[ID] = new_labels

        if idx % 1000 == 0:
            print(idx, img_path)

    np.save(out_path, labels_info)

if __name__ == '__main__':
    out_path = './data/labels_info.npy'
    generate_labels_info(out_path)

