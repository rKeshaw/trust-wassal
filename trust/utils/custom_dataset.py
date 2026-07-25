import numpy as np
import os
import torch
import torchvision
from sklearn import datasets
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import PIL.Image as Image
from .utils import *
np.random.seed(42)
torch.manual_seed(42)

class DataHandler_MNIST(Dataset):
    """
    Data Handler to load MNIST dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    """

    def __init__(self, X, Y=None, select=True, use_test_transform=False):
        """
        Constructor
        """
        self.select = select
        self.use_test_transform=use_test_transform
        self.training_gen_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.test_gen_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if not self.select:
            self.X = X
            self.targets = Y
        else:
            self.X = X

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.targets[index]
            x = Image.fromarray(x)
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            if(x.shape[0]==1): x = torch.repeat_interleave(x, 3, 0)
            y=y.long()
            return (x, y.long())

        else:
            x = self.X[index]
            x = Image.fromarray(x)
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            if(x.shape[0]==1): x = torch.repeat_interleave(x, 3, 0)
            return x

    def __len__(self):
        return len(self.X)

class DataHandler_CIFAR10(Dataset):
    """
    Data Handler to load CIFAR10 dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    """

    def __init__(self, X, Y=None, select=True, use_test_transform = False):
        """
        Constructor
        """
        self.select = select
        if(use_test_transform):
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        if not self.select:
            self.X = X
            self.targets = Y
            self.transform = transform
        else:
            self.X = X
            self.transform = transform

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.targets[index]
            x = Image.fromarray(x)
            x = self.transform(x)
            return (x, y)

        else:
            x = self.X[index]
            x = Image.fromarray(x)
            x = self.transform(x)
            return x

    def __len__(self):
        return len(self.X)

class DataHandler_SVHN(Dataset):
    """
    Data Handler to load SVHN dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    """

    def __init__(self, X, Y=None, select=True, use_test_transform=False):
        """
        Constructor
        """
        self.select = select
        self.use_test_transform=use_test_transform
        self.training_gen_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.test_gen_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std
        if not self.select:
            self.X = X
            self.targets = Y
        else:
            self.X = X

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.targets[index]
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            return (x, y)

        else:
            x = self.X[index]
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            if self.use_test_transform:
                x = self.test_gen_transform(x)
            else:
                x = self.training_gen_transform(x)
            return x

    def __len__(self):
        return len(self.X)

class DataHandler_UTKFace(Dataset):
    """
    Data Handler to load UTKFace dataset.
    This class extends :class:`torch.utils.data.Dataset` to handle 
    loading data even without labels

    Parameters
    ----------
    X: numpy array
        Data to be loaded   
    y: numpy array, optional
        Labels to be loaded (default: None)
    select: bool
        True if loading data without labels, False otherwise
    """
    def __init__(self, X, Y=None, select=True, use_test_transform = False):
        """
        Constructor
        """
        self.select = select
        if(use_test_transform):
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std
        else:
            transform = transforms.Compose([transforms.RandomCrop(200, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet mean/std
        if not self.select:
            self.X = X
            self.targets = Y
            self.transform = transform
        else:
            self.X = X
            self.transform = transform

    def __getitem__(self, index):
        if not self.select:
            x, y = self.X[index], self.targets[index]
            x = Image.fromarray(np.transpose(x, (1,2,0)))
            x = self.transform(x)
            return (x, y)

        else:
            x = self.X[index]
            x = Image.fromarray(x)
            x = self.transform(x)
            return x

    def __len__(self):
        return len(self.X)
    
class DuplicateChannels(object):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return torch.repeat_interleave(pic.unsqueeze(1), 3, 1).float()

    def __repr__(self):
        return self.__class__.__name__ + '()'

def getOODtargets(targets, sel_cls_idx, ood_cls_id):
    
    ood_targets = []
    targets_list = list(targets)
    for i in range(len(targets_list)):
        if(targets_list[i] in list(sel_cls_idx)):
            ood_targets.append(targets_list[i])
        else:
            ood_targets.append(ood_cls_id)
    print("num ood samples: ", ood_targets.count(ood_cls_id))
    return torch.Tensor(ood_targets)
    
def create_ood_data(dset_name, fullset, testset, split_cfg, num_cls, augVal):
    
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    test_idx = [] 
    selected_classes = np.array(list(range(split_cfg['num_cls_idc'])))
    for i in range(num_cls): #all_classes
        if(dset_name=="mnist"):
            full_idx_class = list(torch.where(torch.Tensor(fullset.targets.float()) == i)[0].cpu().numpy())
        else:    
            full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        if(i in selected_classes):
            if(dset_name=="mnist"):
                test_idx_class = list(torch.where(torch.Tensor(testset.targets.float()) == i)[0].cpu().numpy())
            else:
                test_idx_class = list(torch.where(torch.Tensor(testset.targets) == i)[0].cpu().numpy())
            test_idx += test_idx_class
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_idc_train'], replace=False))
            train_idx += class_train_idx
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_idc_val'], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_idc_lake'], replace=False))
        else:
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_ood_train'], replace=False)) #always 0
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_ood_val'], replace=False)) #Only for CG ood val has samples
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_ood_lake'], replace=False)) #many ood samples in lake
    
        if(augVal and (i in selected_classes)): #augment with samples only from the imbalanced classes
            train_idx += class_val_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx
    if(dset_name=="mnist"):
        train_set = SubsetWithTargetsSingleChannel(fullset, train_idx, torch.Tensor(fullset.targets.float())[train_idx])
        val_set = SubsetWithTargetsSingleChannel(fullset, val_idx, torch.Tensor(fullset.targets.float())[val_idx])
        lake_set = SubsetWithTargetsSingleChannel(fullset, lake_idx, getOODtargets(torch.Tensor(fullset.targets.float())[lake_idx], selected_classes, split_cfg['num_cls_idc']))
        test_set = SubsetWithTargetsSingleChannel(testset, test_idx, torch.Tensor(testset.targets.float())[test_idx])
    else:
        train_set = SubsetWithTargets(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
        val_set = SubsetWithTargets(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
        lake_set = SubsetWithTargets(fullset, lake_idx, getOODtargets(torch.Tensor(fullset.targets)[lake_idx], selected_classes, split_cfg['num_cls_idc']))
        test_set = SubsetWithTargets(testset, test_idx, torch.Tensor(testset.targets)[test_idx])

    return train_set, val_set, test_set, lake_set, selected_classes

def create_class_imb(dset_name, fullset, test_set,split_cfg, num_cls, augVal):
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    test_idx=[]
    if(dset_name=="mnist"): selected_classes=np.array([5,8])
    else: selected_classes = np.random.choice(np.arange(num_cls), size=split_cfg['num_cls_imbalance'], replace=False) #classes to imbalance
    for i in range(num_cls): #all_classes
        if(dset_name=="mnist"):
            full_idx_class = list(torch.where(torch.Tensor(fullset.targets.float()) == i)[0].cpu().numpy())
        elif(dset_name=="svhn"):
            full_idx_class = list(torch.where(torch.Tensor(fullset.labels) == i)[0].cpu().numpy())
        else:    
            full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        if(i in selected_classes):
            if(dset_name=="mnist"):
                test_idx_class = list(torch.where(torch.Tensor(test_set.targets.float()) == i)[0].cpu().numpy())
            else:
                test_idx_class = list(torch.where(torch.Tensor(test_set.targets) == i)[0].cpu().numpy())
            test_idx += test_idx_class
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_imbclass_train'], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_val'], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_imbclass_lake'], replace=False))
        else:
            class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_class_train'], replace=False))
            remain_idx = list(set(full_idx_class) - set(class_train_idx))
            class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_val'], replace=False))
            remain_idx = list(set(remain_idx) - set(class_val_idx))
            class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_lake'], replace=False))
    
        train_idx += class_train_idx
        if(augVal and (i in selected_classes)): #augment with samples only from the imbalanced classes
            train_idx += class_val_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx
    if(dset_name=="mnist"):
        train_set = SubsetWithTargetsSingleChannel(fullset, train_idx, torch.Tensor(fullset.targets.float())[train_idx])
        val_set = SubsetWithTargetsSingleChannel(fullset, val_idx, torch.Tensor(fullset.targets.float())[val_idx])
        lake_set = SubsetWithTargetsSingleChannel(fullset, lake_idx, torch.Tensor(fullset.targets.float())[lake_idx])
        test_set = SubsetWithTargetsSingleChannel(fullset, test_idx, torch.Tensor(fullset.targets.float())[test_idx])
    elif(dset_name=="svhn"):
        train_set = SubsetWithTargets(fullset, train_idx, torch.Tensor(fullset.labels)[train_idx])
        val_set = SubsetWithTargets(fullset, val_idx, torch.Tensor(fullset.labels)[val_idx])
        lake_set = SubsetWithTargets(fullset, lake_idx, torch.Tensor(fullset.labels)[lake_idx])        
    else:
        train_set = SubsetWithTargets(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
        val_set = SubsetWithTargets(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
        lake_set = SubsetWithTargets(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx])
    
    return train_set, val_set, test_set,lake_set, selected_classes

def getDuplicateData(dset_name, fullset, split_cfg):
    num_rep=split_cfg['num_rep']
    if(dset_name=="mnist"):
        # X  = np.resize(fullset.data.float().cpu().numpy(), (len(fullset),32,32))
        X  = fullset.data.numpy()
        y = torch.from_numpy(np.array(fullset.targets.float()))
    elif(dset_name == "svhn"):
        X = fullset.data
        y = torch.from_numpy(np.array(fullset.labels))
    else:
        X = fullset.data
        y = torch.from_numpy(np.array(fullset.targets))
    X_tr = X[:split_cfg['train_size']]
    y_tr = y[:split_cfg['train_size']]
    X_unlabeled = X[split_cfg['train_size']:len(X)-split_cfg['val_size']]
    y_unlabeled = y[split_cfg['train_size']:len(X)-split_cfg['val_size']]
    X_val = X[len(X)-split_cfg['val_size']:]
    y_val = y[len(X)-split_cfg['val_size']:]
    X_unlabeled_rep = np.repeat(X_unlabeled[:split_cfg['lake_subset_repeat_size']], num_rep, axis=0)
    y_unlabeled_rep = np.repeat(y_unlabeled[:split_cfg['lake_subset_repeat_size']], num_rep, axis=0)
    assert((X_unlabeled_rep[0]==X_unlabeled_rep[num_rep-1]).all())
    assert((y_unlabeled_rep[0]==y_unlabeled_rep[num_rep-1]).all())
    X_unlabeled_rep = np.concatenate((X_unlabeled_rep, X_unlabeled[split_cfg['lake_subset_repeat_size']:split_cfg['lake_size']]), axis=0)
    y_unlabeled_rep = torch.from_numpy(np.concatenate((y_unlabeled_rep, y_unlabeled[split_cfg['lake_subset_repeat_size']:split_cfg['lake_size']]), axis=0))
    if(dset_name=="mnist"):
        train_set = DataHandler_MNIST(X_tr, y_tr, False)
        lake_set = DataHandler_MNIST(X_unlabeled_rep, y_unlabeled_rep, False)
        val_set = DataHandler_MNIST(X_val, y_val, False)
    elif(dset_name=="svhn"):
        train_set = DataHandler_SVHN(X_tr, y_tr, False)
        lake_set = DataHandler_SVHN(X_unlabeled_rep, y_unlabeled_rep, False)
        val_set = DataHandler_SVHN(X_val, y_val, False)      
    else:
        train_set = DataHandler_CIFAR10(X_tr, y_tr, False)
        lake_set = DataHandler_CIFAR10(X_unlabeled_rep, y_unlabeled_rep, False)
        val_set = DataHandler_CIFAR10(X_val, y_val, False)
    return X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, lake_set

def getVanillaData(dset_name, fullset, split_cfg):
    if(dset_name=="mnist"):
        # X  = np.resize(fullset.data.float().cpu().numpy(), (len(fullset),32,32))
        X  = fullset.data.numpy()
        y = torch.from_numpy(np.array(fullset.targets.float()))
    else:
        X = fullset.data
        y = torch.from_numpy(np.array(fullset.targets))
    X_tr = X[:split_cfg['train_size']]
    y_tr = y[:split_cfg['train_size']]
    X_unlabeled = X[split_cfg['train_size']:len(X)-split_cfg['val_size']]
    y_unlabeled = y[split_cfg['train_size']:len(X)-split_cfg['val_size']]
    X_val = X[len(X)-split_cfg['val_size']:]
    y_val = y[len(X)-split_cfg['val_size']:]
    if(dset_name=="mnist"):
        train_set = DataHandler_MNIST(X_tr, y_tr, False)
        lake_set = DataHandler_MNIST(X_unlabeled, y_unlabeled, False)
        val_set = DataHandler_MNIST(X_val, y_val, False)
    else:
        train_set = DataHandler_CIFAR10(X_tr, y_tr, False)
        lake_set = DataHandler_CIFAR10(X_unlabeled, y_unlabeled, False)
        val_set = DataHandler_CIFAR10(X_val, y_val, False)
    return X_tr, y_tr, X_val, y_val, X_unlabeled[:split_cfg['lake_size']], y_unlabeled[:split_cfg['lake_size']], train_set, val_set, lake_set

def create_perclass_imb(dset_name, fullset, split_cfg, num_cls, augVal):
    np.random.seed(42)
    train_idx = []
    val_idx = []
    lake_idx = []
    selected_classes=split_cfg['sel_cls_idx']
    for i in range(num_cls): #all_classes
        full_idx_class = list(torch.where(torch.Tensor(fullset.targets) == i)[0].cpu().numpy())
        class_train_idx = list(np.random.choice(np.array(full_idx_class), size=split_cfg['per_class_train'][i], replace=False))
        remain_idx = list(set(full_idx_class) - set(class_train_idx))
        class_val_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_val'][i], replace=False))
        remain_idx = list(set(remain_idx) - set(class_val_idx))
        class_lake_idx = list(np.random.choice(np.array(remain_idx), size=split_cfg['per_class_lake'][i], replace=False))

        train_idx += class_train_idx
        if(augVal and (i in selected_classes)): #augment with samples only from the imbalanced classes
            train_idx += class_val_idx
        val_idx += class_val_idx
        lake_idx += class_lake_idx
    train_set = SubsetWithTargets(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
    val_set = SubsetWithTargets(fullset, val_idx, torch.Tensor(fullset.targets)[val_idx])
    lake_set = SubsetWithTargets(fullset, lake_idx, torch.Tensor(fullset.targets)[lake_idx])

    return train_set, val_set, lake_set, selected_classes

def load_dataset_custom(datadir, dset_name, feature, split_cfg, augVal=False, dataAug=True):
    """
    Loads a common dataset with additional options to create class imbalances, out-of-distribution classes, and redundancies.

    Parameters
    ----------
    datadir : string
        The root directory in which the data is stored (or should be downloaded)
    dset_name : string
        The name of the dataset. This should be one of 'cifar10', 'mnist', 'svhn', 'cifar100', 'breast-density'.
    feature : string
        The modification that should be applied to the dataset. This should be one of 'classimb', 'ood', 'duplicate', 'vanilla'
    split_cfg : dict
        Contains information relating to the dataset splits that should be created. Some of the keys for this dictionary are as follows:
            'per_imbclass_train': int
                The number of examples in the train set for each imbalanced class (classimb)
            'per_imbclass_val': int
                The number of examples in the validation set for each imbalanced class (classimb)
            'per_imbclass_lake': int
                The number of examples in the lake set for each imbalanced class (classimb)
            'per_class_train': int
                The number of examples in the train set for each balanced class (classimb)
            'per_class_val': int
                The number of examples in the validation set for each balanced class (classimb)
            'per_class_lake': int
                The number of examples in the lake set for each balanced class (classimb)
            'sel_cls_idx': list
                A list of classes that are affected by class imbalance. (classimb)
            'train_size': int
                The size of the train set (vanilla, duplicate)
            'val_size': int
                The size of the validation set (vanilla, duplicate)
            'lake_size': int
                The size of the lake set (vanilla, duplicate)
            'num_rep': int
                The number of times to repeat a selection in the lake set (duplicate)
            'lake_subset_repeat_size': int
                The size of the repeated selection in the lake set (duplicate)
            'num_cls_imbalance': int
                The number of classes to randomly affect by class imbalance. (classimb)
            'num_cls_idc': int
                The number of in-distribution classes to keep (ood)
            'per_idc_train': int
                The number of in-distribution examples to keep in the train set per class (ood)
            'per_idc_val': int
                The number of in-distribution examples to keep in the validation set per class (ood)
            'per_idc_lake': int
                The number of in-distribution examples to keep in the lake set per class (ood)
            'per_ood_train': int
                The number of OOD examples to keep in the train set per class (ood)
            'per_ood_val': int
                The number of OOD examples to keep in the validation set per class (ood)
            'per_ood_lake': int
                The number of OOD examples to keep in the lake set per class (ood)         
    augVal : bool, optional
        If True, the train set will also contain affected classes from the validation set. The default is False.
    dataAug : bool, optional
        If True, the all but the test set will be affected by random cropping and random horizontal flip. The default is True.

    Returns
    -------
    tuple
        Returns a train set, validation set, test set, lake set, and number of classes. Amount of returned items depends on specific configuration.
        Each set is an instance of torch.utils.data.Dataset
    """
    
    if(not(os.path.exists(datadir))):
        os.mkdir(datadir)

    if(dset_name=="cifar10"):
        num_cls=10
        cifar_test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        if(dataAug):
            cifar_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            cifar_transform = cifar_test_transform
        
        fullset = torchvision.datasets.CIFAR10(root=datadir, train=True, download=True, transform=cifar_transform)
        test_set = torchvision.datasets.CIFAR10(root=datadir, train=False, download=True, transform=cifar_test_transform)
        if(feature=="classimb"):
            if("sel_cls_idx" in split_cfg):
                train_set, val_set, lake_set, imb_cls_idx = create_perclass_imb(dset_name, fullset, split_cfg, num_cls, augVal)
            else:
                train_set, val_set, test_set,lake_set, imb_cls_idx = create_class_imb(dset_name, fullset, test_set,split_cfg, num_cls, augVal)
            print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
        if(feature=="ood"):
            train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(dset_name, fullset, test_set, split_cfg, num_cls, augVal)
            print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
            return train_set, val_set, test_set, lake_set, ood_cls_idx, split_cfg['num_cls_idc']
        if(feature=="vanilla"): 
            X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set = getVanillaData(dset_name, fullset, split_cfg)
            print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, num_cls

        if(feature=="duplicate"):
            X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, lake_set = getDuplicateData(dset_name, fullset, split_cfg)
            print("CIFAR-10 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, num_cls

    if(dset_name=="mnist"):
        num_cls=10
        mnist_test_transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if(dataAug):
            mnist_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        else:
            mnist_transform = mnist_test_transform
        fullset = torchvision.datasets.MNIST(root=datadir, train=True, download=True, transform=mnist_transform)
        test_set = torchvision.datasets.MNIST(root=datadir, train=False, download=True, transform=mnist_test_transform)
        #test_set=SubsetWithTargets(fullset, train_idx, torch.Tensor(fullset.targets)[train_idx])
        #fullset.data = torch.repeat_interleave(fullset.data.unsqueeze(1), 3, 1).float()
        #test_set.data = torch.repeat_interleave(test_set.data.unsqueeze(1), 3, 1).float()
        if(feature=="classimb"):
            if("sel_cls_idx" in split_cfg):
                train_set, val_set, lake_set, imb_cls_idx = create_perclass_imb(dset_name, fullset, split_cfg, num_cls, augVal)
            else:
                train_set, val_set, test_set, lake_set, imb_cls_idx = create_class_imb(dset_name, fullset, test_set,split_cfg, num_cls, augVal)
            print("MNIST Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
        if(feature=="ood"):
            train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(dset_name, fullset, test_set, split_cfg, num_cls, augVal)
            print("MNIST Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
            return train_set, val_set, test_set, lake_set, ood_cls_idx, split_cfg['num_cls_idc']
        if(feature=="vanilla"): 
            X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set = getVanillaData(dset_name, fullset, split_cfg)
            print("MNIST Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, num_cls

        if(feature=="duplicate"):
            X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, lake_set = getDuplicateData(dset_name, fullset, split_cfg)
            print("MNIST Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, num_cls

    if(dset_name=="svhn"):
        num_cls=10
        SVHN_test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        if(dataAug):
            SVHN_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        else:
            SVHN_transform = SVHN_test_transform
        
        fullset = torchvision.datasets.SVHN(root=datadir, split="train", download=True, transform=SVHN_transform)
        test_set = torchvision.datasets.SVHN(root=datadir, split="test", download=True, transform=SVHN_test_transform)

        #svhn has labels and we need to rename it targets
        fullset.targets = fullset.labels
        test_set.targets = test_set.labels
        if(feature=="classimb"):
            if("sel_cls_idx" in split_cfg):
                train_set, val_set, lake_set, imb_cls_idx = create_perclass_imb(dset_name, fullset, split_cfg, num_cls, augVal)
            else:
                train_set, val_set, lake_set, imb_cls_idx = create_class_imb(dset_name, fullset, split_cfg, num_cls, augVal)
            print("SVHN Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
        if(feature=="ood"):
            train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(dset_name, fullset, test_set, split_cfg, num_cls, augVal)
            print("SVHN Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
            return train_set, val_set, test_set, lake_set, ood_cls_idx, split_cfg['num_cls_idc']
        if(feature=="vanilla"): 
            X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set = getVanillaData(dset_name, fullset, split_cfg)
            print("SVHN Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, num_cls

        if(feature=="duplicate"):
            X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, lake_set = getDuplicateData(dset_name, fullset, split_cfg)
            print("SVHN Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, num_cls

    if(dset_name=="cifar100"):
        num_cls=100
        cifar100_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        fullset = torchvision.datasets.CIFAR100(root=datadir, train=True, download=True, transform=cifar100_transform)
        test_set = torchvision.datasets.CIFAR100(root=datadir, train=False, download=True, transform=cifar100_transform)
        if(feature=="classimb"):
            if("sel_cls_idx" in split_cfg):
                train_set, val_set, lake_set, imb_cls_idx = create_perclass_imb(dset_name, fullset, split_cfg, num_cls, augVal)
            else:
                train_set, val_set, lake_set, imb_cls_idx = create_class_imb(dset_name, fullset, split_cfg, num_cls, augVal)
            print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls
        if(feature=="ood"):
            train_set, val_set, test_set, lake_set, ood_cls_idx = create_ood_data(dset_name, fullset, test_set, split_cfg, num_cls, augVal)
            print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set), "Test set: ", len(test_set))
            return train_set, val_set, test_set, lake_set, ood_cls_idx, num_cls
        
        if(feature=="vanilla"):
            X_tr, y_tr, X_val, y_val, X_unlabeled, y_unlabeled, train_set, val_set, lake_set = getVanillaData(dset_name, fullset, split_cfg)
            print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, num_cls

        if(feature=="duplicate"):
            X_tr, y_tr, X_val, y_val, X_unlabeled_rep, y_unlabeled_rep, train_set, val_set, lake_set = getDuplicateData(dset_name, fullset, split_cfg)
            print("CIFAR-100 Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, num_cls
            
    # --- STL-10 support: load from stl10_raw_channel_first.npz and use full unlabeled pool as lake ---
    if dset_name == "stl10":
        import numpy as _np
        from torch.utils.data import Dataset as _TDataset, Subset as _Subset
    
        num_cls = 10
        npz_path = os.path.join(datadir, "stl10_raw_channel_first.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"STL-10 npz file not found: {npz_path}")
    
        dataz = _np.load(npz_path)
        # expected keys: X_train, y_train, X_test, y_test, X_unlabeled, y_unlabeled (y_unlabeled maybe placeholders)
        X_train_np = dataz["X_train"]
        y_train_np = dataz["y_train"]
        X_test_np = dataz["X_test"]
        y_test_np = dataz["y_test"]
        X_unlabeled_np = dataz["X_unlabeled"]
        y_unlabeled_np = dataz.get("y_unlabeled", None)  # may be placeholders or absent
    
        class NPZDataset(_TDataset):
            def __init__(self, X, y=None, transform=None):
                import torch as _torch
                self.X = X
                # keep targets as numpy array for compatibility with existing code
                self.targets = _np.array(y) if y is not None else None
                self.transform = transform
    
            def __len__(self):
                return len(self.X)
    
            def __getitem__(self, idx):
                import torch as _torch
                x = self.X[idx]  # expected channel-first np array (C,H,W)
                x_t = _torch.from_numpy(x).float()
                if self.transform:
                    x_t = self.transform(x_t)
                if self.targets is None:
                    y = -1
                else:
                    y = int(self.targets[idx])
                return x_t, y
    
        # Full datasets
        full_train_ds = NPZDataset(X_train_np, y_train_np)
        test_set = NPZDataset(X_test_np, y_test_np)
        lake_full_ds = NPZDataset(X_unlabeled_np, y_unlabeled_np)
    
        # Build train and val splits from full_train_ds using split_cfg (per_class_train/per_class_val)
        # We will sample per-class from X_train (deterministic ordering). If you want randomness, shuffle cls_lists.
        from collections import defaultdict
        cls_indices = defaultdict(list)
        for idx, lab in enumerate(full_train_ds.targets):
            cls_indices[int(lab)].append(idx)
    
        train_idxs = []
        val_idxs = []
        for cls in range(num_cls):
            cls_list = cls_indices[cls]
            # optionally shuffle:
            # import random; random.shuffle(cls_list)
            n_train = int(split_cfg["per_class_train"][cls]) if "per_class_train" in split_cfg else 20
            n_val = int(split_cfg["per_class_val"][cls]) if "per_class_val" in split_cfg else 10
            if len(cls_list) < (n_train + n_val):
                # if available samples are fewer than requested, take what is available and warn
                take = cls_list
                train_take = take[:n_train] if len(take) >= n_train else take[: max(0, len(take) - n_val)]
                val_take = take[len(train_take) : len(train_take) + n_val]
                print(f"WARNING: class {cls} has only {len(cls_list)} train images; requested {n_train}+{n_val}.")
            else:
                train_take = cls_list[:n_train]
                val_take = cls_list[n_train : n_train + n_val]
            train_idxs += train_take
            val_idxs += val_take
    
        # Build Subset objects and attach .targets attributes expected by the rest of the code
        train_set = _Subset(full_train_ds, train_idxs)
        train_set.targets = full_train_ds.targets[train_idxs]
    
        val_set = _Subset(full_train_ds, val_idxs)
        val_set.targets = full_train_ds.targets[val_idxs]
    
        # lake_set: use full unlabeled pool (all indices)
        lake_indices = list(range(len(lake_full_ds)))
        lake_set = _Subset(lake_full_ds, lake_indices)
        # if y_unlabeled exists and used for analysis, attach targets; else set to -1
        # if lake_full_ds.targets is not None:
        #     lake_set.targets = lake_full_ds.targets[lake_indices]
        # else:
        #     import numpy as _np
        #     lake_set.targets = _np.array([-1] * len(lake_indices))
        
        # lake_set: use the remainder of the Labeled training data that was not used for train or val
        all_labeled_indices = set(range(len(full_train_ds)))
        used_indices = set(train_idxs + val_idxs)
        lake_indices = sorted(list(all_labeled_indices - used_indices))

        lake_set = _Subset(full_train_ds, lake_indices)
        lake_set.targets = full_train_ds.targets[lake_indices]
    
        sel_cls_idx = list(range(num_cls))
    
        print("STL-10 Custom dataset stats: Train size:", len(train_set),
              "Val size:", len(val_set), "Lake size:", len(lake_set), "Test size:", len(test_set))
        return train_set, val_set, test_set, lake_set, sel_cls_idx, num_cls

    
    if(dset_name=="breast_density"):
        num_cls=4
        input_size=224
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        fullset = datasets.ImageFolder(os.path.join(datadir, 'train'), data_transforms['train'])
        test_set = datasets.ImageFolder(os.path.join(datadir, 'test'), data_transforms['test'])
        if(feature=="classimb"):
            train_set, val_set, lake_set, imb_cls_idx = create_perclass_imb(dset_name, fullset, split_cfg, num_cls, augVal)
            print("Breast-density Custom dataset stats: Train size: ", len(train_set), "Val size: ", len(val_set), "Lake size: ", len(lake_set))
            return train_set, val_set, test_set, lake_set, imb_cls_idx, num_cls