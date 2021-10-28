import torch
import time
import numpy as np

from CLAPPVision.utils import logger
from CLAPPVision.vision.arg_parser import arg_parser
from CLAPPVision.vision.models import load_vision_model
from CLAPPVision.vision.data import get_dataloader


def train(opt, model, train_loader, optimizer):
    total_step = len(train_loader)
    model.module.switch_calc_loss(True)

    print_idx = 100

    starttime = time.time()
    cur_train_module = opt.train_module

    for epoch in range(opt.start_epoch, opt.num_epochs + opt.start_epoch):

        loss_epoch = [0 for i in range(opt.model_splits)]
        loss_updates = [1 for i in range(opt.model_splits)]

        # loop over batches in train_loader
        for step, (img, label) in enumerate(train_loader):

            if step % print_idx == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Training Block: {}, Time (s): {:.1f}".format(
                        epoch + 1,
                        opt.num_epochs + opt.start_epoch,
                        step,
                        total_step,
                        cur_train_module,
                        time.time() - starttime,
                    )
                )

            starttime = time.time()

            model_input = img.to(opt.device)
            label = label.to(opt.device)

            # forward pass through whole model (loop over modules within model forward)
            loss, loss_gated, _, _, accuracy = model(model_input, label, n=cur_train_module)
            loss = torch.mean(loss, 0) # take mean over outputs of different GPUs
            loss_gated = torch.mean(loss_gated, 0) # take mean over outputs of different GPUs
            accuracy = torch.mean(accuracy, 0)

            if cur_train_module != opt.model_splits and opt.model_splits > 1:
                raise ValueError("Training intermediate modules is not tested!")
                # loss = loss[cur_train_module].unsqueeze(0)

            # loop through the losses of the modules and do gradient descent
            for idx in range(len(loss)):
                if len(loss) == 1 and opt.model_splits != 1:
                    idx = cur_train_module

                model.zero_grad()

                if idx == len(loss) - 1: # last module
                    loss[idx].backward()
                else:
                    # select loss or loss_gated for gradient descent
                    loss_for_grads = loss_gated[idx] if model.module.opt.feedback_gating else loss[idx]
                    loss_for_grads.backward(retain_graph=True)

                optimizer[idx].step()
                if opt.predict_module_num=='-1' or opt.predict_module_num=='both':
                    if idx != 0:
                        optimizer[idx-1].step() # to update lower (feature) layer
                if opt.predict_module_num=='-1b' and not opt.skip_upper_c_update:
                    if idx != len(loss) - 1:
                        optimizer[idx+1].step() # to update upper (context) layer

                # We still output normal (ungated) loss for printing and plotting
                print_loss = loss[idx].item()
                if opt.asymmetric_W_pred: 
                    print_loss *= 0.5 # loss is double in that case but gradients are still the same -> print the corresponding values
                print_acc = accuracy[idx].item()
                if step % print_idx == 0:
                    print("\t \t Loss: \t \t {:.4f}".format(print_loss))
                    if opt.loss == 1:
                        print("\t \t Accuracy: \t \t {:.4f}".format(print_acc))

                loss_epoch[idx] += print_loss
                loss_updates[idx] += 1

        logs.append_train_loss([x / loss_updates[idx] for idx, x in enumerate(loss_epoch)])
        logs.create_log(model, epoch=epoch, optimizer=optimizer)

if __name__ == "__main__":

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)
    opt.training_dataset = "unlabeled"

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.device.type != "cpu":
        torch.backends.cudnn.benchmark = True

    # load model
    model, optimizer = load_vision_model.load_model_and_optimizer(opt)

    logs = logger.Logger(opt)

    train_loader, _, supervised_loader, _, test_loader, _ = get_dataloader.get_dataloader(
        opt
    )

    if opt.loss == 1:
        train_loader = supervised_loader

    try:
        # Train the model
        train(opt, model, train_loader, optimizer)

    except KeyboardInterrupt:
        print("Training got interrupted, saving log-files now.")

    logs.create_log(model)
