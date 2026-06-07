import os 
import pickle
from torch.utils import data 
from dataset.cityscapes import CityscapesDataSet, CityscapesTrainInform, CityscapesValDataSet, CityscapesTestDataSet
from dataset.camvid import CamVidDataSet, CamVidValDataSet, CamVidTrainInform, CamVidTestDataSet
from dataset.voc import VOCSeg
from dataset.udd import UDDPatchDataset
import albumentations as A
import getpass

if getpass.getuser() == 'lmh':
    VOC_path = '/data/ubuntu/lmh/VOC'   #for fu's server
    UDD_path = '/storage/lmh/UDD/UDD6/preprocessed' # for fu's server
elif getpass.getuser() == 'met4physics':
    VOC_path = '/mnt/d/VOC' # for workplace
    UDD_path = '/mnt/d/UDD/UDD6/preprocessed' # for workplace
elif getpass.getuser() == 'wyl':
    VOC_path = '/home/wyl/VOC'
    UDD_path = '/home/wyl/UDD/UDD6/preprocessed'
elif getpass.getuser() == 'user22':
    UDD_path = '/data1/user22/UDD/UDD6/preprocessed'
elif getpass.getuser() == 'root':
    UDD_path = '/root/autodl-tmp/UDD/UDD6/preprocessed'


def build_dataset_train(dataset, input_size, batch_size, train_type, random_scale, random_mirror, num_workers):
    data_dir = os.path.join('./dataset/', dataset)
    dataset_list = dataset + '_trainval_list.txt'
    train_data_list = os.path.join(data_dir, dataset + '_' + train_type + '_list.txt')
    val_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    if dataset not in ['voc', 'udd']:
        # inform_data_file collect the information of mean, std and weigth_class
        if not os.path.isfile(inform_data_file):
            print("%s is not found" % (inform_data_file))
            if dataset == "cityscapes":
                dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=dataset_list,
                                                    inform_data_file=inform_data_file)
            elif dataset == 'camvid':
                dataCollect = CamVidTrainInform(data_dir, 11, train_set_file=dataset_list,
                                                inform_data_file=inform_data_file)
            else:
                raise NotImplementedError(
                    "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)

            datas = dataCollect.collectDataAndSave()
            if datas is None:
                print("error while pickling data. Please check.")
                exit(-1)
        else:
            print("find file: ", str(inform_data_file))
            datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "cityscapes": 
                                         # def __init__(self, root='', list_path='', max_iters=None,
        trainLoader = data.DataLoader(
            CityscapesDataSet('/root/autodl-tmp/cityscapes', train_data_list, crop_size=input_size, scale=random_scale,
                              mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(                  
            CityscapesValDataSet('/root/autodl-tmp/cityscapes', val_data_list, f_scale=1, mean=datas['mean']),
            batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True,
            drop_last=True)

        return datas, trainLoader, valLoader

    elif dataset == "camvid":

        trainLoader = data.DataLoader(
            CamVidDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                          mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            CamVidValDataSet(data_dir, val_data_list, f_scale=1, mean=datas['mean']),
            batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)

        return datas, trainLoader, valLoader
    
    elif dataset == 'voc':
        
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Resize(input_size[0], input_size[1]),
        ])
        test_transform = A.Compose([
            A.Resize(input_size[0], input_size[1])
        ])

        trainLoader = data.DataLoader(
            VOCSeg(root=VOC_path, year='2012', image_set='train', download=False, transforms=train_transform),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            VOCSeg(root=VOC_path, year='2012', image_set='val', download=False, transforms=test_transform),
            batch_size=20, shuffle=False, num_workers=num_workers, pin_memory=True)

        return None, trainLoader, valLoader
    
    elif dataset == 'udd':
        
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Resize(input_size[0], input_size[1]),
        ])
        test_transform = A.Compose([
            A.Resize(input_size[0], input_size[1])
        ])

        trainLoader = data.DataLoader(
            UDDPatchDataset(txt_file=os.path.join(UDD_path, 'train_patches.txt'), transforms=train_transform),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            UDDPatchDataset(txt_file=os.path.join(UDD_path, 'val_patches.txt'), transforms=test_transform),
            batch_size=20, shuffle=False, num_workers=num_workers, pin_memory=True)

        return None, trainLoader, valLoader


def build_dataset_test(dataset, num_workers, none_gt=False, batch_size=8):#if test on validation set, set none_gt to False
    data_dir = os.path.join('./dataset/', dataset)
    dataset_list = dataset + '_trainval_list.txt'
    test_data_list = os.path.join(data_dir, dataset + '_test' + '_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    if dataset not in ['voc', 'udd']:
        # inform_data_file collect the information of mean, std and weigth_class
        if not os.path.isfile(inform_data_file):
            print("%s is not found" % (inform_data_file))
            if dataset == "cityscapes":
                dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=dataset_list,
                                                    inform_data_file=inform_data_file)
            elif dataset == 'camvid':
                dataCollect = CamVidTrainInform(data_dir, 11, train_set_file=dataset_list,
                                                inform_data_file=inform_data_file)
            else:
                raise NotImplementedError(
                    "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)
            
            datas = dataCollect.collectDataAndSave()
            if datas is None:
                print("error while pickling data. Please check.")
                exit(-1)
        else:
            print("find file: ", str(inform_data_file))
            datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "cityscapes":
        # for cityscapes, if test on validation set, set none_gt to False
        # if test on the test set, set none_gt to True
        if none_gt: 
            testLoader = data.DataLoader(
                CityscapesTestDataSet(data_dir, test_data_list, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        else: 
            test_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
            testLoader = data.DataLoader(
                CityscapesValDataSet(data_dir, test_data_list, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, testLoader

    elif dataset == "camvid":

        testLoader = data.DataLoader(
            CamVidValDataSet(data_dir, test_data_list, mean=datas['mean']),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, testLoader
    
    elif dataset == 'voc':
        
        test_transform = A.Compose([
            A.Resize(400, 400)
        ])


        valLoader = data.DataLoader(
            VOCSeg(root=VOC_path, year='2012', image_set='val', download=False, transforms=test_transform),
            batch_size=10, shuffle=False, num_workers=num_workers, pin_memory=True)

        return None, valLoader
    
    elif dataset == 'udd':
        
        test_transform = A.Compose([
            A.Resize(400, 400)
        ])


        valLoader = data.DataLoader(
            UDDPatchDataset(txt_file=os.path.join(UDD_path, 'val_patches.txt'), transforms=test_transform),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        return None, valLoader

