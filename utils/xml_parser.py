from xml.etree.ElementTree import parse

class XMLParser(object):
    def __init__(self, xml_path):
        self.file_name = ''
        self.width = 0
        self.height = 0
        self.objects = []
        
        tree = parse(xml_path)
        root = tree.getroot()
        
        self.file_name = root.find('filename').text
        self.width = int(root.find('size').find('width').text)
        self.height = int(root.find('size').find('height').text)
        
        objs = root.findall('object')
        
        for obj in objs:
            class_name = obj.find('name').text
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            
            self.objects.append([class_name, xmin, ymin, xmax, ymax])
