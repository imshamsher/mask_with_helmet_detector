import xmltodict
import cv2

import os

conversions_dir = "./yolo_format_annotations/"

try:
    os.makedirs(conversions_dir)
except :
    pass


#opening the xml file in read mode
with open("annotations.xml","r") as xml_obj:
    #coverting the xml data to Python dictionary
    context = xmltodict.parse(xml_obj.read())
    #closing the file
    xml_obj.close()

with open("object.names","w") as names:
    names.write("both_helmet_and_mask\n")
    names.write("not_both_helmet_and_mask")

image_data= context["annotations"]["image"]

for img_d in image_data:
    print(img_d)

    image_name = img_d["@id"] + ".jpg"

    boxes = img_d["box"]

    img = cv2.imread("../task2/dataset/images/" + img_d["@id"] + ".jpg")

    rows, cols, _ = img.shape
    yolo_marks = []
    for b in boxes:

        if b["@label"] == "head":
            p1 = (int(float(b["@xtl"])) , int(float(b["@ytl"])))
            p2 = (int(float(b["@xbr"])), int(float(b["@ybr"])))
            attri = b["attribute"]

            abs_c_x = int((p1[0] + p2[0]) / 2)
            abs_c_y = int((p1[1] + p2[1]) / 2)
            abs_w = p2[0] - p1[0]
            abs_h = p2[1] - p1[1]

            relative_center_x = float(abs_c_x / cols)
            relative_center_y = float(abs_c_y / rows)
            relative_width = float(abs_w / cols)
            relative_height = float(abs_h / rows)


            helmet = False
            mask = False
            label_index = 0
            for a in attri:
                if a["@name"] == "mask" and a["#text"] == "yes":
                    label_index = 0
                    mask = True
                if a["@name"] == "has_safety_helmet" and a["#text"] == "yes":
                    label_index = 1
                    helmet = True


            if helmet and mask:
                cv2.rectangle(img, p1,p2, (0,255,0),2)

            else:
                cv2.rectangle(img, p1,p2, (0,0,255),2)

            yolo_marks.append(str(label_index) + " " +
                              str(relative_center_x) + " " +
                              str(relative_center_y) + " " +
                              str(relative_width) + " " +
                              str(relative_height))




    cv2.imshow("image", img)
    cv2.waitKey(1)

    if len(yolo_marks) != 0:
        with open(conversions_dir + '/' + image_name[0:-3] + 'txt', 'w') as f:
            pass
        for m in yolo_marks:
            with open(conversions_dir + '/' + image_name[0:-3] + 'txt', 'a') as f:
                f.write(m + '\n')