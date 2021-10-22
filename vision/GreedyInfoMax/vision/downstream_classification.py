import torch
import numpy as np
import time
import os
import code

from GreedyInfoMax.vision.data import get_dataloader
from GreedyInfoMax.vision.arg_parser import arg_parser
from GreedyInfoMax.vision.models import load_vision_model
from GreedyInfoMax.utils import logger, utils


def train_logistic_regression(opt, context_model, classification_model, train_loader):
    total_step = len(train_loader)
    classification_model.train()

    starttime = time.time()

    # No randomness in cifar training -> save reps and go without forward through context model!
    if opt.dataset == "cifar10" or opt.dataset == "cifar100":
        reps = [] 

    for epoch in range(opt.num_epochs):
        epoch_acc1 = 0
        epoch_acc5 = 0

        loss_epoch = 0

        for step, (img, target) in enumerate(train_loader):
            
            classification_model.zero_grad()
            
            if (opt.dataset == "cifar10" or opt.dataset == "cifar100") and (epoch != 0):
                z = reps[step]
            else:
                model_input = img.to(opt.device)

                if opt.model_type == 2:  # fully supervised training
                    _, _, z = context_model(model_input)
                else:
                    with torch.no_grad():
                        _, _, _, z, _ = context_model(model_input, target, n=opt.module_num)
                    z = z.detach() # double security that no gradients go to representation learning part of model

                if opt.dataset == "cifar10" or opt.dataset == "cifar100":
                    reps.append(z)

            prediction = classification_model(z)

            target = target.to(opt.device)
            loss = criterion(prediction, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy
            acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
            epoch_acc1 += acc1
            epoch_acc5 += acc5

            sample_loss = loss.item()
            loss_epoch += sample_loss

            if step % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc5: {:.4f}, Loss: {:.4f}".format(
                        epoch + 1,
                        opt.num_epochs,
                        step,
                        total_step,
                        time.time() - starttime,
                        acc1,
                        acc5,
                        sample_loss,
                    )
                )
                starttime = time.time()

        if opt.validate:
            # validate the model - in this case, test_loader loads validation data
            val_acc1, _ , val_loss = test_logistic_regression(
                opt, context_model, classification_model, test_loader
            )
            logs.append_val_loss([val_loss])

        print("Overall accuracy for this epoch: ", epoch_acc1 / total_step)
        logs.append_train_loss([loss_epoch / total_step])
        logs.create_log(
            context_model,
            epoch=epoch,
            classification_model=classification_model,
            accuracy=epoch_acc1 / total_step,
            acc5=epoch_acc5 / total_step,
        )


def test_logistic_regression(opt, context_model, classification_model, test_loader):
    total_step = len(test_loader)
    context_model.eval()
    classification_model.eval()

    starttime = time.time()

    loss_epoch = 0
    epoch_acc1 = 0
    epoch_acc5 = 0

    for step, (img, target) in enumerate(test_loader):

        model_input = img.to(opt.device)

        with torch.no_grad():
            _, _, _, z, _ = context_model(model_input, target, n=opt.module_num)

        z = z.detach()

        prediction = classification_model(z)

        target = target.to(opt.device)
        loss = criterion(prediction, target)

        # calculate accuracy
        acc1, acc5 = utils.accuracy(prediction.data, target, topk=(1, 5))
        epoch_acc1 += acc1
        epoch_acc5 += acc5

        sample_loss = loss.item()
        loss_epoch += sample_loss

        if step % 10 == 0:
            print(
                "Step [{}/{}], Time (s): {:.1f}, Acc1: {:.4f}, Acc5: {:.4f}, Loss: {:.4f}".format(
                    step, total_step, time.time() - starttime, acc1, acc5, sample_loss
                )
            )
            starttime = time.time()

    print("Testing Accuracy: ", epoch_acc1 / total_step)
    return epoch_acc1 / total_step, epoch_acc5 / total_step, loss_epoch / total_step


if __name__ == "__main__":

    opt = arg_parser.parse_args()

    add_path_var = "linear_model"

    arg_parser.create_log_path(opt, add_path_var=add_path_var)
    opt.training_dataset = "train"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # load pretrained model
    # cannot switch opt.reduced_patch_pooling = False here because otherwise W_preds sizes don't match
    context_model, _ = load_vision_model.load_model_and_optimizer(
        opt, reload_model=True, calc_loss=False
    )
    context_model.module.switch_calc_loss(False)
    if opt.reduced_patch_pooling:
        print("ATTENTION: the option --reduced_patch_pooling is only active for GIM/CLAPP training: " 
            "For downstream classification (from lower layers), activations are still pooled over the whole patch (in the downstream classifier). "
            "Otherwise change line below this warning and change the number of in_channels accordingly.")
        for module in context_model.module.encoder:
            module.patch_average_pool_out_dim = 1
    

    # model_type=2 is supervised model which trains entire architecture; otherwise just extract features
    if opt.model_type != 2:
        context_model.eval()
        
    if opt.module_num==-1:
        print("CAREFUL! Training classifier directly on input image! Model is ignored and returns the (flattened) input images!")

    _, _, train_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt)

    classification_model = load_vision_model.load_classification_model(opt)

    if opt.model_type == 2:
        params = list(context_model.parameters()) + list(classification_model.parameters())
    else:
        params = classification_model.parameters()

    optimizer = torch.optim.Adam(params)
    criterion = torch.nn.CrossEntropyLoss()

    logs = logger.Logger(opt)

    try:
        # Train the model
        train_logistic_regression(opt, context_model, classification_model, train_loader)

        # Test the model
        acc1, acc5, _ = test_logistic_regression(
            opt, context_model, classification_model, test_loader
        )

    except KeyboardInterrupt:
        print("Training got interrupted")

    logs.create_log(
        context_model,
        classification_model=classification_model,
        accuracy=acc1,
        acc5=acc5,
        final_test=True,
    )
    torch.save(
        context_model.state_dict(), os.path.join(opt.log_path, "context_model.ckpt")
    )

    np.save(os.path.join(opt.model_path, "classification_results_values_"+str(opt.dataset)+".npy"), 
            np.array([acc1, acc5]))
    L = ["Test top1 classification accuracy: "+str(acc1)+"\n",
        "Test top5 classification accuracy: "+str(acc5)+"\n"]
    f = open(os.path.join(opt.model_path, "classification_results_"+str(opt.dataset)+".txt"), "w")
    f.writelines(L)
    f.close()
