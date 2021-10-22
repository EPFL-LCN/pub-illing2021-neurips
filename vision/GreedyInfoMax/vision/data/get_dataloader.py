import torch
import torchvision.transforms as transforms
import torchvision
import os
import numpy as np
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from IPython import embed


def get_dataloader(opt):
    if opt.dataset == "stl10":
        print("load STL-10 dataset...")
        train_loader, train_dataset, supervised_loader, supervised_dataset, test_loader, test_dataset = get_stl10_dataloader(
            opt
        )
    elif opt.dataset == "cifar10" or opt.dataset == "cifar100":
        train_loader, train_dataset, supervised_loader, supervised_dataset, test_loader, test_dataset = get_cifar_dataloader(
            opt
        )
        # train_loader and train_dataset are None in this case!
    else:
        raise Exception("Invalid option")

    # embed()
    # raise Exception()
    return (
        train_loader,
        train_dataset,
        supervised_loader,
        supervised_dataset,
        test_loader,
        test_dataset,
    )


def get_stl10_dataloader(opt):
    base_folder = os.path.join(opt.data_input_dir, "stl10_binary")

    aug = {
        "stl10": {
            "randcrop": opt.random_crop_size,
            "flip": True,
            "resize": False,
            "pad": False,
            "grayscale": opt.grayscale,
            "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
            "std": [0.2683, 0.2610, 0.2687],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
    }
    transform_train = transforms.Compose(
        [get_transforms(eval=False, aug=aug["stl10"])]
    )
    transform_valid = transforms.Compose(
        [get_transforms(eval=True, aug=aug["stl10"])]
    )

    unsupervised_dataset = torchvision.datasets.STL10(
        base_folder,
        split="unlabeled",
        transform=transform_train,
        download=opt.download_dataset,
    ) #set download to True to get the dataset

    train_dataset = torchvision.datasets.STL10(
        base_folder, split="train", transform=transform_train, download=opt.download_dataset
    )

    test_dataset = torchvision.datasets.STL10(
        base_folder, split="test", transform=transform_valid, download=opt.download_dataset
    )

    # default dataset loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size_multiGPU, shuffle=True, num_workers=16
    )

    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=True,
        num_workers=16,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, num_workers=16
    )

    # create train/val split
    if opt.validate:
        print("Use train / val split")

        if opt.training_dataset == "train":
            dataset_size = len(train_dataset)
            train_sampler, valid_sampler = create_validation_sampler(dataset_size)

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=opt.batch_size_multiGPU,
                sampler=train_sampler,
                num_workers=16,
            )

        elif opt.training_dataset == "unlabeled":
            dataset_size = len(unsupervised_dataset)
            train_sampler, valid_sampler = create_validation_sampler(dataset_size)

            unsupervised_loader = torch.utils.data.DataLoader(
                unsupervised_dataset,
                batch_size=opt.batch_size_multiGPU,
                sampler=train_sampler,
                num_workers=16,
            )

        else:
            raise Exception("Invalid option")

        # overwrite test_dataset and _loader with validation set
        test_dataset = torchvision.datasets.STL10(
            base_folder,
            split=opt.training_dataset,
            transform=transform_valid,
            download=opt.download_dataset,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size_multiGPU,
            sampler=valid_sampler,
            num_workers=16,
        )

    else:
        print("Use (train+val) / test split")

    return (
        unsupervised_loader,
        unsupervised_dataset,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )


def create_validation_sampler(dataset_size):
    # Creating data indices for training and validation splits:
    validation_split = 0.2
    shuffle_dataset = True

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler

# only exists in v0.9.9
# class Sharpen:
#     """Sharpen image after upsampling with interpolation."""
#     def __call__(self, x):
#         return TF.adjust_sharpness(x, 2.)

def get_transforms(eval=False, aug=None):
    trans = []

    if aug["resize"]:
        trans.append(transforms.Resize(aug["resize_size"]))

    if aug["pad"]:
        trans.append(transforms.Pad(aug["pad_size"], fill=0, padding_mode='constant'))
    
    if aug["randcrop"] and not eval:
        trans.append(transforms.RandomCrop(aug["randcrop"]))

    if aug["randcrop"] and eval:
        trans.append(transforms.CenterCrop(aug["randcrop"]))

    if aug["flip"] and not eval:
        trans.append(transforms.RandomHorizontalFlip())

    if aug["grayscale"]:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    elif aug["mean"]:
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
    else:
        trans.append(transforms.ToTensor())

    trans = transforms.Compose(trans)
    return trans


def get_cifar_dataloader(opt):
    cor_factor_mean = 0.06912513 # correction factors: STL-10 normalisation lead to these residual mean and std -> has to be adapted to get same input distribution
    cor_factor_std = 0.95930314
    if opt.dataset == "cifar10":
        print("load cifar10 dataset...")
        base_folder = os.path.join(opt.data_input_dir, "cifar10_binary")
        bw_mean = 0.47896898 - cor_factor_mean * 0.2392343 / cor_factor_std
        bw_std = 0.2392343 / cor_factor_std
    elif opt.dataset == "cifar100":
        print("load cifar100 dataset...")
        base_folder = os.path.join(opt.data_input_dir, "cifar100_binary")
        bw_mean = 0.48563015 - cor_factor_mean * 0.25072286 / cor_factor_std
        bw_std = 0.25072286 / cor_factor_std

    aug = {
        "cifar": {
            "resize": False,
            "resize_size": 64, # 96
            "pad": False,
            "pad_size": 16,
            "randcrop": False, #opt.random_crop_size,
            "flip": False,
            "grayscale": opt.grayscale,
            "bw_mean": [bw_mean],
            "bw_std": [bw_std],
        }
    }
    # mean and std found as:
    # x = np.concatenate([np.asarray(im) for (im, t) in supervised_loader]); np.mean(x); np.std(x)
    # CIFAR10
    # for vanilla 32 x 32 input: "bw_mean": [0.47896898], "bw_std": [0.2392343]
    # for resize_size: 96 and randcrop: "bw_mean": [0.470379], "bw_std": [0.2249]
    # for resize_size: 64 without randcrop: "bw_mean": [0.4798809], "bw_std": [0.23278822]
    # for pad: True and pad_size: 16: "bw_mean": [0.11974239], "bw_std": [0.23942184]
    # CIFAR100
    # for vanilla 32 x 32 input: "bw_mean": [0.48563015], "bw_std": [0.25072286]

    transform_train = transforms.Compose(
        [get_transforms(eval=False, aug=aug["cifar"])]
    )
    transform_valid = transforms.Compose(
        [get_transforms(eval=True, aug=aug["cifar"])]
    )

    if opt.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            base_folder, train=True, transform=transform_train, download=opt.download_dataset
        )
        test_dataset = torchvision.datasets.CIFAR10(
            base_folder, train=False, transform=transform_valid, download=opt.download_dataset
        )
    elif opt.dataset == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            base_folder, train=True, transform=transform_train, download=opt.download_dataset
        )
        test_dataset = torchvision.datasets.CIFAR100(
            base_folder, train=False, transform=transform_valid, download=opt.download_dataset
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, num_workers=16
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, num_workers=16
    )

    return (
        None,
        None,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )