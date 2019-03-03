import torch
import torch.utils.data
import numpy as np
import dataset_reader
from torch.utils.data import DataLoader

class raw_detection_dataset(torch.utils.data.Dataset):
    def __init__(self,root="",image_type="raw",image_shape=(512,512),label_set=["bicycle","car","person"],):
        reader=dataset_reader.data_reader(root,image_type,image_shape,label_set)
        print("reading dataset")
        self.length=reader.get_length()
        self.data=[]
        self.box=[]
        self.label=[]
        for i in range(0,self.length):
            data=reader.get_data(i)
            self.data.append(data)
            print(data.shape)
            clss,bndbox=reader.get_label(i)
            self.label.append(clss)
            self.box.append(bndbox)
        print("end reading")

    def __getitem__(self,index):
        image=self.data[index].astype(np.float32)
        box=np.array(self.box[index]).astype(np.int64)
        label=np.array(self.label[index]).astype(np.int64)
        image=torch.from_numpy(image)
        box=torch.from_numpy(box)
        label=torch.from_numpy(label)
        return image,torch.cat((box,label),1)

    def __len__(self):
        return self.length


if __name__ == '__main__':
    root="/media/jiazizhou/9D64F10C1ABED33A/PASCALRAW/PASCALRAW"
    dataset=raw_detection_dataset(root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
