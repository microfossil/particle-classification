import os
from collections import OrderedDict, namedtuple
from pathlib import Path
import lxml.etree as ET
import pandas as pd
import numpy as np

ClsInfo = namedtuple('ClsInfo', 'cls score val_by project')

total_projects = 0



def parse_xml(xml_filename, data):
    project = ET.parse(xml_filename).getroot()

    if data is None:
        data = OrderedDict()

    images_xml = project.find('images')
    if images_xml is None:
        return data
    total_images = 0
    for i, image_xml in enumerate(images_xml.iter('image')):
        total_images += 1
        relfile = image_xml.find('source').find('filename').text
        if os.path.isabs(relfile):
            absfile = relfile
        else:
            absfile = os.path.abspath(os.path.join(os.path.dirname(xml_filename), relfile))
        if os.path.isfile(absfile) is False:
            continue

        cls_names = []
        cls_scores = []
        cls_base = image_xml.find('classifications')
        for cls_val in cls_base.iter('classification'):
            cls_names.append(cls_val.find('code').text)
            cls_scores.append(float(cls_val.find('value').text))
        if absfile not in data:
            data[absfile] = []
        val_by = image_xml.find('val_by')
        if val_by is None:
            val_by = ""
        else:
            val_by = val_by.text
        info = ClsInfo(cls_names[np.argmax(cls_scores)],
                       np.max(cls_scores),
                       val_by,
                       xml_filename)
        data[absfile].append(info)

    print(total_images)

    return data

    # for taxon_xml in project.find('taxons').iter('taxon'):
    #     if taxon_xml.find('isClass').text == 'true':
    #         cls_labels.append(taxon_xml.find('code').text)
    # cls_labels = sorted(cls_labels)
    #
    # df = pd.DataFrame({'filenames': filenames, 'cls': cls})
    # for label in cls_labels:
    #     filenames_dict[label] = df.filenames[df.cls == label]
    # return filenames_dict


SOURCE_DIR = r"C:\Users\rossm\Documents\Data"

xmls = Path(SOURCE_DIR).rglob("*.xml")
data = None
for xml in xmls:
    print(xml)
    with open(xml, "r") as pfn:
        pfn.readline()
        line = pfn.readline()
        if line.startswith("<project"):
            data = parse_xml(str(xml), data)
            total_projects += 1
            if total_projects > 10:
                break

counts = {}
for key, value in data.items():
    for info in value:
        if info.cls not in counts:
            counts[info.cls] = 0
        counts[info.cls] += 1
        print(f"{key},{info.cls},{info.score},{info.val_by},{info.project}")

print(counts)

with open("info.csv", "w") as fp:
    fp.write("Filename,Label,Annotator,Project\n")
    for key, value in data.items():
        for info in value:
            if info.cls not in counts:
                counts[info.cls] = 0
            counts[info.cls] += 1
            fp.write(f"{key},{info.cls},{info.val_by},{info.project}\n")