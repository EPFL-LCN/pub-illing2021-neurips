from optparse import OptionGroup

def parse_train_args(parser):
    group = OptionGroup(parser, "Training options")
    group.add_option(
        "--learning_rate", 
        type="float", 
        default=2e-4, 
        help="Learning rate (for ADAM optimiser)"
    )
    group.add_option(
        "--weight_decay", 
        type="float", 
        default=0., 
        help="weight decay or l2-penalty on weights (for ADAM optimiser, default = 0., i.e. no l2-penalty)"
    )
    group.add_option(
        "--prediction_step",
        type="int",
        default=5,
        help="(Number of) Time steps to predict into future",
    )
    group.add_option(
        "--gradual_prediction_steps",
        action="store_true",
        default=False,
        help="Increase number of time steps (to predict into future) module by module. This is meant to be used with 6 modules",
    )
    group.add_option(
        "--reduced_patch_pooling",
        action="store_true",
        default=False,
        help="Reduce adaptive average pooling of patch encodings. This means that some spatial information is kept." 
            "The dimension of context and target vectors grow accordingly. This is meant to be used with 6 modules.",
    )
    group.add_option(
        "--negative_samples",
        type="int",
        default=16,
        help="Number of negative samples to be used for training",
    )
    group.add_option(
        "--current_rep_as_negative",
        action="store_true",
        default=False,
        help="Use the current feature vector ('context' at time t as opposed to predicted time step t+k) itself as/for sampling the negative sample",
    )
    group.add_option(
        "--sample_negs_locally",
        action="store_true",
        default=False,
        help="Sample neg. samples from batch but within same location in image, i.e. no shuffling across locations",
    )
    group.add_option(
        "--sample_negs_locally_same_everywhere",
        action="store_true",
        default=False,
        help="Extension of --sample_negs_locally_same_everywhere (must be True). No shuffling across locations and same sample (from batch) for all locations. I.e. negative sample is simply a new input without any scrambling",
    )
    group.add_option(
        "--either_pos_or_neg_update",
        action="store_true",
        default=False,
        help="Randomly chose to do either pos or neg update in Hinge loss. --negative_samples should be 1. Only used with --current_rep_as_negative True",
    )
    group.add_option(
        "--patch_size",
        type="int",
        default=16,
        help="Encoding patch size. Use single integer for same encoding size for all modules (default=16)",
    )
    group.add_option(
        "--increasing_patch_size",
        action="store_true",
        default=False,
        help="Boolean: start with patch size 4 and increase by factors 2 per module until max. patch size = --patch_size (e.g. 16)",
    )
    group.add_option(
        "--random_crop_size",
        type="int",
        default=64,
        help="Size of the random crop window. Use single integer for same size for all modules (default=64)",
    )
    group.add_option(
        "--inpatch_prediction",
        action="store_true",
        default=False,
        help="Boolean: change CPC task to smaller scale prediction (within patch -> smaller receptive field) by extra unfolding ",
    )
    group.add_option(
        "--inpatch_prediction_limit",
        type="int",
        default=2,
        help="Number of module below which inpatch prediction is applied (if inpatch prediction is active) (default=2, i.e. modules 0 and 1 are doing inpatch prediction)",
    )
    group.add_option(
        "--feedback_gating",
        action="store_true",
        default=False,
        help="Boolean: use feedback from higher layers to gate lower layer plasticity",
    )
    group.add_option(
        "--gating_av_over_preds",
        action="store_true",
        default=False,
        help="Boolean: average feedback gating (--feedback_gating) from higher layers over different prediction steps ('k')",
    )
    group.add_option(
        "--contrast_mode",
        type="str",
        default="multiclass",
        help="decides whether constrasting with neg. examples is done at once 'mutliclass' "
                "or one at a time with (and then averaged) with CE 'binary', BCE 'logistic' or 'hinge' loss",
    )
    group.add_option(
        "--detach_c",
        action="store_true",
        default=False,
        help="Boolean whether the gradient of the context c should be dropped (detached)",
    )
    group.add_option(
        "--encoder_type",
        type="str",
        default="resnet",
        help="Select the encoder type: resnet or vgg_like",
    )
    group.add_option(
        "--inference_recurrence",
        type="int",
        default=0,
        help="recurrence (on the module level) during inference (before evaluating loss):"
        "0 - no recurrence"
        "1 - lateral recurrence within layer"
        "2 - feedback recurrence"
        "3 - both, lateral and feedback recurrence",
    )
    group.add_option(
        "--recurrence_iters",
        type="int",
        default=5,
        help="number of iterations for inference recurrence (without recurrence, --inference_recurrence == 0, it is set to 0) ",
    )
    group.add_option(
        "--model_splits",
        type="int",
        default=3,
        help="Number of individually trained modules that the original model should be split into "
             "options: 1 (normal end-to-end backprop) or 3 (default used in experiments of paper)",
    )
    group.add_option(
        "--train_module",
        type="int",
        default=3,
        help="Index of the module to be trained individually (0-2), "
        "or training network as one (3)",
    )
    group.add_option(
        "--predict_module_num",
        type="str",
        default="same",
        help="Option whether W should predict activities in the same module ('same', default), "
             "one module below with first module predicting same module ('-1'),"
             "both ('both') or"
             "one module below with last module predicting same module ('-1b')",
    )
    group.add_option(
        "--extra_conv",
        action="store_true",
        default=False,
        help="Boolian whether extra convolutional layer too increase rec. field size (with downsampling, i.e. stride > 1)"
             "is used to decode activity before avg-pooling and contrastive loss",
    )
    group.add_option(
        "--asymmetric_W_pred",
        action="store_true",
        default=False,
        help="Boolean: solve weight transport in W_pred by using two distinct W_pred(1,2) and splitting the score:"
            "Loss(u) -> Loss1(u1) + Loss2(u2) for both, pos. and neg. samples, with"
            "u = z*W_pred*c -> u1 = drop_grad(z)*W_pred1*c, u2 = z*W_pred2*drop_grad(c)",
    )
    group.add_option(
        "--freeze_W_pred",
        action="store_true",
        default=False,
        help="Boolean whether the k prediction weights W_pred (W_k in ContrastiveLoss) are frozen (require_grad=False).",
    )
    group.add_option(
        "--unfreeze_last_W_pred",
        action="store_true",
        default=False,
        help="Boolean whether the k prediction weights W_pred of the last module should be unfrozen.",
    )
    group.add_option(
        "--skip_upper_c_update",
        action="store_true",
        default=False,
        help="Boolean whether extra update in upper (context) layer is skipped. Consider this when predicting lower modules",
    )
    group.add_option(
        "--no_gamma",
        action="store_true",
        default=False,
        help="Boolean whether gamma (factor which sets the opposite sign of the update for pos and neg samples) is set to 1. i.e. third factor omitted in learning rule",
    )
    group.add_option(
        "--no_pred",
        action="store_true",
        default=False,
        help="Boolean whether Wpred * c is set to 1 (no prediction). i.e. fourth factor omitted in learning rule",
    )

    parser.add_option_group(group)
    return parser
