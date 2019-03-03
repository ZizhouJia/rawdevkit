import numpy as np
import os
import utils
import cv2

class data_reader:
    def __init__(self,root,image_type="raw",image_shape=None,label_set=["bicycle","car","person"]):
        self.image_shape=image_shape
        self.image_type=image_type
        self.root=root
        self.image_names=[]
        self.label_set=label_set
        path=os.path.join(self.root,"annotations")
        for name in os.listdir(path):
            if(".xml" in name):
                self.image_names.append(name[:-4])


    def get_length(self):
        return len(self.image_names)


    def get_label(self,index):
        if(index>=self.get_length()):
            return None
        path=os.path.join(self.root,"annotations")
        path=os.path.join(path,self.image_names[index]+".xml")
        clss,bounding_boxs=utils.read_xml_and_get_info(path)
        for i in range(0,len(clss)):
            clss[i]=self.label_set.index(clss[i])
        for i in range(0,len(bounding_boxs)):
            if(self.image_shape is not None):
                box=bounding_boxs[i]
                box[0]=int(float(box[0])*self.image_shape[1]/600)
                box[2]=int(float(box[2])*self.image_shape[1]/600)
                box[1]=int(float(box[1])*self.image_shape[0]/400)
                box[3]=int(float(box[3])*self.image_shape[0]/400)
                bounding_boxs[i]=box
        return clss,bounding_boxs


    def get_data(self,index):
        if(index>=self.get_length()):
            return None
        if(self.image_type=="raw"):
            path=os.path.join(self.root,"original/raw")
            path=os.path.join(path,self.image_names[index]+".nef")
            raw_data=utils.read_raw(path)
            if(self.image_shape is not None):
                raw_data=utils.resize_raw(raw_data,self.image_shape)
            return np.expand_dims(raw_data, axis=2)
        if(self.image_type=="rgb"):
            path=os.path.join(self.root,"original/jpg")
            path=os.path.join(path,self.image_names[index]+".jpg")
            image=cv2.imread(path)
            if(self.image_shape is not None):
                image=cv2.resize(image,(self.image_shape[1],self.image_shape[0]))
            return image


if __name__ == '__main__':
    reader=data_reader(root="/media/jiazizhou/9D64F10C1ABED33A/PASCALRAW/PASCALRAW",image_type="rgb")
    print(reader.get_length())
    print(reader.get_label(0))
    print(reader.get_data(1).shape)
