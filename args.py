import argparse
import network

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--default_path", type=str, default="/data1/sdi/CPNKD-result/",
                        help="path to results")
    parser.add_argument("--current_time", type=str, default=None,
                        help="results images folder name (default: current time)")
    # Tensorboard options
    parser.add_argument("--Tlog_dir", type=str, default=None,
                        help="path to tensorboard log")
    parser.add_argument("--save_log", action='store_true', default=True, 
                        help="save log to default path (default: False)")
    # Model options
    available_models = sorted(name for name in network.model.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.model.__dict__[name])
                             )
    parser.add_argument("--t_model", type=str, default='unet_rgb', choices=available_models,
                        help='model name (default: Unet RGB)')
    parser.add_argument("--s_model", type=str, default='unet_rgb', choices=available_models,
                        help='model name (default: Unet RGB)')
    parser.add_argument("--t_model_params", type=str, default='/',
                        help="pretrained teacher network params (default: '/')")
    # DeeplabV3+ options
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp (default: False)")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16, 32, 64],
                        help="output stride (default: 16)")
    parser.add_argument("--t_separable_conv", action='store_true', default=False,
                        help="teacher model: apply separable conv to decoder and aspp (default: False)")
    parser.add_argument("--t_output_stride", type=int, default=16, choices=[8, 16, 32, 64],
                        help="teacher model: output stride (default: 16)")
    # Dataset options
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers (default: 4)")
    parser.add_argument("--data_root", type=str, default="/data1/sdi/datasets/",
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default="CPN_six",
                        help='Name of dataset (default: CPN_six)')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes (default: 2)")
    parser.add_argument("--is_rgb", action='store_false', default=True,
                        help="choose True: RGB, False: grey (default: True)")
    # Augmentation options
    parser.add_argument("--resize", default=(496, 468))
    parser.add_argument("--crop_size", default=(512, 448))
    parser.add_argument("--val_crop_size", default=(512, 448))
    parser.add_argument("--scale_factor", type=float, default=5e-1)
    # Train options
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--gpus", type=str, default='6,7',
                        help="GPU IDs (default: 6,7)")
    parser.add_argument("--total_itrs", type=int, default=2500,
                        help="epoch number (default: 2.5k)")
    parser.add_argument("--lr", type=float, default=1e-1,
                        help="learning rate (default: 1e-1)")
    parser.add_argument("--loss_type", type=str, default='dice_loss',
                        help="criterion (default: dice loss)")
    parser.add_argument("--optim", type=str, default='SGD',
                        help="optimizer (default: SGD)")
    parser.add_argument("--lr_policy", type=str, default='step',
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=100, 
                        help="step size (default: 1000)")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')

    parser.add_argument("--alpha", type=float, default=0.5,
                        help="alpha for KD loss (default: 0.5)")
    parser.add_argument("--T", type=float, default=3,
                        help="temperature in KD loss (default: 3)")
    parser.add_argument("--std", type=float, default=0.01,
                        help="sigma in gaussian perturbation (default: 0.01)")
    # Early stop options
    parser.add_argument("--patience", type=int, default=100,
                        help="Number of epochs with no improvement after which training will be stopped (default: 100)")
    parser.add_argument("--delta", type=float, default=0.001,
                        help="Minimum change in the monitored quantity to qualify as an improvement (default: 0.001)")
    # Validate options
    parser.add_argument("--val_interval", type=int, default=1,
                        help="epoch interval for eval (default: 1)")
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validate (default: 4)')
    # Outcome options
    parser.add_argument("--save_model", action='store_true', default=False,
                        help="save best model param to \"./best_param\" (default: False)")
    parser.add_argument("--save_ckpt", type=str, default=None,
                        help="save best model param to \"./best_param\"")
    parser.add_argument("--val_results", action='store_true', default=False,
                        help='save validate segmentation results to \"./val_results\" (default: False)')
    parser.add_argument("--val_results_dir", type=str, default=None,
                        help="save segmentation results to \"./results\"")
    # Model checkpoint options
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False,
                        help="restore state from reserved params (defaults: false)")

    # Log-in info
    parser.add_argument("--login_dir", type=str, default="/login.json",
                        help="path to user log-in info json file (default: /login.json)")
    parser.add_argument("--cur_work_server", type=int, default=5,
                        help="current working server (default: 5)")
    # Run Demo
    parser.add_argument("--run_demo", action='store_true', default=False)

    return parser

if __name__ == "__main__":
    
    import utils

    opts = get_argparser().parse_args()
    jsummary = {}
    for key, val in vars(opts).items():
        jsummary[key] = val
    utils.save_dict_to_json(d=jsummary, json_path='/data1/sdi/CPNKD/utils/sample/opts_sample.json')

    pram = utils.Params('/data1/sdi/CPNKD/utils/sample/opts_sample.json')

    print(type(pram.separable_conv)) # bool
    print(type(pram.num_classes)) # int
    print(type(pram.weight_decay)) # float
    print(type(pram.ckpt)) # str

    pram.update(json_path='/data1/sdi/CPNKD/utils/sample/mlog.json')
    utils.save_dict_to_json(pram.__dict__, '/data1/sdi/CPNKD/utils/sample/out.json')