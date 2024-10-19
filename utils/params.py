import argparse
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def args_parser():
    parser = argparse.ArgumentParser(description="Personalized Federated Learning")

    # Simulation args
    parser.add_argument("--num-steps", type=int, default=30, help="number of FL rounds")
    parser.add_argument("--num-nodes", type=int, default=20, help="number of simulated nodes (n clients)")
    
    # Data args
    parser.add_argument("--data-path", type=str, default="data/REFIT", help="dir path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=24*6)
    parser.add_argument("--forecast-horizon", type=int, default=24)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--nb-features", type=int, default=1)
    parser.add_argument("--valid-set-size", type=float, default=0.15)
    parser.add_argument("--test-set-size", type=float, default=0.15)

    # Forecasting training args
    parser.add_argument("--inner-steps", type=int, default=200, help="number of training epochs on each client")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--verbose", type=str2bool, default=False)

    # Forecasting model args
    parser.add_argument("--hid-size", type=float, default=1.0)
    parser.add_argument("--num-stacks", type=int, default=2)
    parser.add_argument("--num-levels", type=int, default=3)
    parser.add_argument("--num-decoder-layer", type=int, default=1)
    parser.add_argument("--concat-len", type=int, default=0)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--single-step-output-One", type=int, default=0)
    parser.add_argument("--positionale", type=str2bool, default=False)
    parser.add_argument("--modified", type=str2bool, default=True)
    parser.add_argument("--RIN", type=str2bool, default=False)
    parser.add_argument("--L1Loss", type=str2bool, default=True)
    parser.add_argument("--decompose", type=str2bool, default=False)

    # FEDORA args
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-5, help="inner weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")

    # General args
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=20, help="eval every X selected epochs") # this is used twice, TODO rename for main or for forecasting model
    parser.add_argument("--save-path", type=str, default="results", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=0, help="seed value")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    print("=" * 50)

    return args
