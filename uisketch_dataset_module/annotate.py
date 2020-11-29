""" Generate Pascal VOC annotation XML file from data """


import xml.etree.ElementTree as xml


class XMLGenerator:
    """Generate Pascal VOC XML annotation files"""

    def __init__(self, filename, width, height, depth=3):
        self.root = self.generate_root(filename, width, height, depth)

    def generate_root(self, filename, width, height, depth):
        """Generate root element of XML file

        Returns:
            Element -- xml ElementTree Element for root
        """

        root = xml.Element("annotation")
        root.set("verified", "yes")

        folder_element = xml.SubElement(root, "folder")
        folder_element.text = "images"

        filename_element = xml.SubElement(root, "filename")
        filename_element.text = f"{filename}.jpg"

        path_element = xml.SubElement(root, "path")
        path_element.text = f"../images/{filename}.jpg"

        source_element = xml.SubElement(root, "source")
        database_element = xml.SubElement(source_element, "database")
        database_element.text = "Generated for UISketch synthetic dataset"

        size_element = xml.SubElement(root, "size")
        width_element = xml.SubElement(size_element, "width")
        width_element.text = str(width)
        height_element = xml.SubElement(size_element, "height")
        height_element.text = str(height)
        depth_element = xml.SubElement(size_element, "depth")
        depth_element.text = str(depth)

        segmented_element = xml.SubElement(root, "segmented")
        segmented_element.text = "0"

        return root

    def add_object(self, name, bndbox):
        """Append object classname and bounding box

        Arguments:
            name {string} -- class name of the object inside bounding box
            bndbox {tuple} -- Tuple of bounding box (xmin, ymin, xmax, ymax)
        """

        xmin, ymin, xmax, ymax = bndbox

        object_element = xml.SubElement(self.root, "object")

        name_element = xml.SubElement(object_element, "name")
        name_element.text = name

        pose_element = xml.SubElement(object_element, "pose")
        pose_element.text = "Unspecified"

        truncated_element = xml.SubElement(object_element, "truncated")
        truncated_element.text = "0"

        difficult_element = xml.SubElement(object_element, "difficult")
        difficult_element.text = "0"

        bndbox_element = xml.SubElement(object_element, "bndbox")
        xmin_element = xml.SubElement(bndbox_element, "xmin")
        xmin_element.text = str(xmin)
        ymin_element = xml.SubElement(bndbox_element, "ymin")
        ymin_element.text = str(ymin)
        xmax_element = xml.SubElement(bndbox_element, "xmax")
        xmax_element.text = str(xmax)
        ymax_element = xml.SubElement(bndbox_element, "ymax")
        ymax_element.text = str(ymax)

    def save_file(self, output_path):
        """Store the XML file

        Arguments:
            output_path {string} -- Annotated XML file path
        """

        tree = xml.ElementTree(self.root)
        with open(output_path, "wb") as file:
            tree.write(file)

    def __str__(self):
        return xml.tostring(self.root, encoding="unicode")
