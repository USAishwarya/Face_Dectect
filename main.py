import os
import xml.etree.ElementTree as ET
import pandas as pd


def xml_to_csv(path):
    xml_list = []
    for xml_file in os.listdir(path):
        if not xml_file.endswith('.xml'):
            continue
        tree = ET.parse(os.path.join(path, xml_file))
        root = tree.getroot()
        filename = root.find('filename').text
        for member in root.findall('object'):
            value = (
                filename,
                int(root.find('size/width').text),
                int(root.find('size/height').text),
                member.find('name').text,
                int(member.find('bndbox/xmin').text),
                int(member.find('bndbox/ymin').text),
                int(member.find('bndbox/xmax').text),
                int(member.find('bndbox/ymax').text)
            )
            xml_list.append(value)
    columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    return pd.DataFrame(xml_list, columns=columns)


annotations_df = xml_to_csv('archive/')
annotations_df.to_csv('labels.csv', index=False)
