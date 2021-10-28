import torch
import numpy as np
import time
import os

from CLAPPVision.vision.data import get_dataloader
from CLAPPVision.vision.arg_parser import arg_parser
from CLAPPVision.vision.models import load_vision_model
from CLAPPVision.utils import logger, utils


def get_loss_and_accuracy(opt, model, data_loader):
    losses = torch.zeros(opt.model_splits)
    accuracies = torch.zeros(opt.model_splits)
    for step, (img, target) in enumerate(data_loader):

        model_input = img.to(opt.device)
        with torch.no_grad():
            loss, _, _, _, accuracy = model(model_input, target, n=opt.module_num)

        loss = torch.mean(loss, 0) # average over GPUs
        accuracy = torch.mean(accuracy, 0)

        for idx in range(opt.model_splits):    
            losses[idx] += loss[idx]
            accuracies[idx] += accuracy[idx]

        if step % 10 == 0:
            print("evaluate batch number ", step, " out of ", len(data_loader))

    return losses / len(data_loader), accuracies / len(data_loader)

if __name__ == "__main__":

    opt = arg_parser.parse_args()
    if opt.loss != 1:
        raise ValueError("--loss keyword is not set to 1 (supervised). This only works for the Supervised Model")

    add_path_var = "linear_model"

    arg_parser.create_log_path(opt, add_path_var=add_path_var)
    opt.training_dataset = "train"

    model, _ = load_vision_model.load_model_and_optimizer(
        opt, reload_model=True, calc_loss=False
    )
    model.module.switch_calc_loss(True)

    _, _, train_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt)
    
    print("Evaluating loss and accuracy on train set...")
    loss_train, acc_train = get_loss_and_accuracy(opt, model, train_loader)
    print("Evaluating loss and accuracy on test set...")
    loss_test, acc_test = get_loss_and_accuracy(opt, model, test_loader)

    np.save(os.path.join(opt.model_path, "classification_results_values.npy"), 
            np.array([loss_train.numpy(), acc_train.numpy(), loss_test.numpy(), acc_test.numpy()]))
    L = ["Training losses for all modules: "+str(loss_train.numpy())+"\n",
        "Training accuracies for all modules: "+str(acc_train.numpy())+"\n",
        "Testing losses for all modules: "+str(loss_test.numpy())+"\n",
        "Testing accuracies for all modules: "+str(acc_test.numpy())+"\n"]
    f = open(os.path.join(opt.model_path, "classification_results.txt"), "w")
    f.writelines(L)
    f.close()
    
    for l in L:
        print(l)

