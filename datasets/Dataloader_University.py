import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image

class Dataloader_University(Dataset):
    def __init__(self,root,transforms,names=['satellite','street','drone','google']):
        super(Dataloader_University).__init__()
        self.transforms_drone_street = transforms['train']
        self.transforms_satellite = transforms['satellite']
        self.root = root
        self.names =  names
        
        # 检查数据路径是否存在
        if not os.path.exists(root):
            raise FileNotFoundError(f"数据根目录 {root} 不存在！请检查数据路径配置。")
        
        #获取所有图片的相对路径分别放到对应的类别中
        #{satelite:{0839:[0839.jpg],0840:[0840.jpg]}}
        dict_path = {}
        available_names = []
        
        for name in names:
            name_path = os.path.join(root, name)
            if os.path.exists(name_path):
                available_names.append(name)
                dict_ = {}
                try:
                    for cls_name in os.listdir(name_path):
                        cls_path = os.path.join(name_path, cls_name)
                        if os.path.isdir(cls_path):
                            img_list = [f for f in os.listdir(cls_path) 
                                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                            img_path_list = [os.path.join(cls_path, img) for img in img_list]
                            if img_path_list:  # 只有当文件夹中有图片时才添加
                                dict_[cls_name] = img_path_list
                    dict_path[name] = dict_
                except Exception as e:
                    print(f"警告：处理 {name} 文件夹时出现错误：{e}")
                    continue
            else:
                print(f"警告：{name_path} 路径不存在，跳过该类别")
        
        if not available_names:
            raise FileNotFoundError(f"在 {root} 中没有找到任何有效的数据类别！")
        
        self.names = available_names
        print(f"成功加载的数据类别：{available_names}")

        #获取设置名字与索引之间的镜像
        # 使用第一个可用的类别来获取类别名称
        first_available = available_names[0]
        cls_names = list(dict_path[first_available].keys())
        cls_names.sort()
        map_dict={i:cls_names[i] for i in range(len(cls_names))}

        self.cls_names = cls_names
        self.map_dict = map_dict
        self.dict_path = dict_path
        self.index_cls_nums = 2
        
        print(f"成功加载 {len(cls_names)} 个类别：{cls_names[:5]}..." if len(cls_names) > 5 else f"成功加载 {len(cls_names)} 个类别：{cls_names}")

    #从对应的类别中抽一张出来
    def sample_from_cls(self,name,cls_num):
        if name not in self.dict_path:
            # 如果指定的名称不存在，使用第一个可用的类别
            name = self.names[0]
            print(f"警告：指定的类别 {name} 不存在，使用 {self.names[0]} 替代")
        
        if cls_num not in self.dict_path[name]:
            print(f"警告：类别 {cls_num} 在 {name} 中不存在")
            # 使用该名称下的第一个可用类别
            available_cls = list(self.dict_path[name].keys())
            if available_cls:
                cls_num = available_cls[0]
            else:
                raise ValueError(f"在 {name} 中没有找到任何类别！")
        
        img_path = self.dict_path[name][cls_num]
        img_path = np.random.choice(img_path,1)[0]
        img = Image.open(img_path)
        return img


    def __getitem__(self, index):
        cls_nums = self.map_dict[index]
        img = self.sample_from_cls("satellite",cls_nums)
        img_s = self.transforms_satellite(img)

        img = self.sample_from_cls("street",cls_nums)
        img_st = self.transforms_drone_street(img)

        img = self.sample_from_cls("drone",cls_nums)
        img_d = self.transforms_drone_street(img)
        return img_s,img_st,img_d,index


    def __len__(self):
        return len(self.cls_names)



class Sampler_University(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source, batchsize=8,sample_num=4):
        self.data_len = len(data_source)
        self.batchsize = batchsize
        self.sample_num = sample_num

    def __iter__(self):
        list = np.arange(0,self.data_len)
        np.random.shuffle(list)
        nums = np.repeat(list,self.sample_num,axis=0)
        return iter(nums)

    def __len__(self):
        return len(self.data_source)


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    img_s,img_st,img_d,ids = zip(*batch)
    ids = torch.tensor(ids,dtype=torch.int64)
    return [torch.stack(img_s, dim=0),ids],[torch.stack(img_st,dim=0),ids], [torch.stack(img_d,dim=0),ids]

if __name__ == '__main__':
    transform_train_list = [
        # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256, 256), interpolation=3),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]


    transform_train_list ={"satellite": transforms.Compose(transform_train_list),
                            "train":transforms.Compose(transform_train_list)}
    datasets = Dataloader_University(root="/home/dmmm/University-Release/train",transforms=transform_train_list,names=['satellite','drone'])
    samper = Sampler_University(datasets,8)
    dataloader = DataLoader(datasets,batch_size=8,num_workers=0,sampler=samper,collate_fn=train_collate_fn)
    for data_s,data_d in dataloader:
        print()


