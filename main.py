import argparse
import time
import logging
import torch
import numpy as np
from pathlib import Path
from src.search import DSE_MC, DSE_PSSE
from src.utils import same_seed, create_mask, logits2prob, load_mat_data

parser = argparse.ArgumentParser(description="Flower Embedded devices")
# User Specified Parameters
parser.add_argument("--task",           type=str,   default="MC",       help="Which task", choices=["MC", "PSSE"], )
parser.add_argument("--search_P",       type=bool,  default=False,      help="")

parser.add_argument("--busses",         type=int,   default=2,          help="bus measurement matrix to load", choices=["2"])
parser.add_argument("--Vmax",           type=float, default=1.3,        help="")
parser.add_argument("--Vmin",           type=float, default=0.7,        help="")
parser.add_argument("--thmin",          type=float, default=-np.pi/2,   help="")
parser.add_argument("--thmax",          type=float, default=np.pi/2,    help="")

parser.add_argument("--seed",           type=int,   default=0,          help="")
parser.add_argument("--device",         type=str,   default="cpu",      help="training and inference device",)
parser.add_argument("--n",              type=int,   default=10,         help="")
parser.add_argument("--rank",           type=int,   default=1,          help="")
parser.add_argument("--sample_prob",    type=float, default=0.1,        help="The probability that each entry can be 1")
parser.add_argument("--coherence",      type=float, default=2.0,        help="coherence of the factor")
parser.add_argument("--e_norm",         type=float, default=1e-18,      help="the maximum norm of error")
parser.add_argument("--loss_type",      type=str,   default="max",      help="", choices=["sum", "max"])
parser.add_argument("--optimizer",      type=str,   default='Adam',     help="", choices=["Adam", "SGD", "L-BFGS"])
parser.add_argument("--iters",          type=int,   default=10000,      help="search iterations")
parser.add_argument("--max_lr",         type=float, default=5e-4,       help="")
parser.add_argument("--min_lr",         type=float, default=5e-5,       help="")
parser.add_argument("--lr_sched",       type=str,   default="cosine",   help="", choices=["static", "cosine", "linear"])
parser.add_argument("--temperature",    type=float, default=1.5,        help="")
parser.add_argument("--max_sq_loss",    type=float, default=-1e-1,      help="")
parser.add_argument("--min_sq_loss",    type=float, default=-1e4,       help="")
parser.add_argument("--search_loops",   type=int,   default=10,         help="")
parser.add_argument("--top_k",          type=int,   default=None,       help="")
parser.add_argument("--M_type",         type=str,   default="STE",      help="")

parser.add_argument("--save_path",      type=Path,  default="./results",help="")


def main():
    args = parser.parse_args()
    
    same_seed(args.seed)
    time_str = time.strftime("%Y%m%d-%H%M%S")
    result_path = args.save_path / time_str
    result_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename = result_path / "output.log",
        format = '%(asctime)s: %(levelname)s: %(message)s', 
        level = logging.INFO
    )
    logging.info("*************** Arguments ***************")
    for arg, val in sorted(vars(args).items()):
        logging.info(f"{arg:<15} : {val}")
    logging.info("*****************************************")
    # Address top-k
    if args.top_k is None:
        top_k = int(args.n * args.n * args.sample_prob)
    else:
        top_k = args.top_k

    device = torch.device(args.device)
    if args.task == 'MC':
        M = torch.zeros((args.n, args.n))#.to_sparse_coo()
        if not args.search_P:
            M = create_mask(
                logits2prob(torch.rand((args.n, args.n)), 'STE'),
                top_k, 
                'STE', 
                M
            )
            top_k = 0
        DSE_MC(
            top_k,
            args.search_loops,
            device,
            args.n,
            args.rank,
            args.coherence,
            args.e_norm,
            f"{args.task}_{args.loss_type}",
            args.optimizer,
            args.iters,
            args.max_lr,
            args.min_lr,
            args.lr_sched,
            args.temperature,
            (args.max_sq_loss, args.min_sq_loss),
            args.M_type,
            M,
            result_path,
        )
    else:
        if args.busses == 2:
            A_np = load_mat_data("data/2bus/2bus_data.mat")
            n = 2 * args.busses - 1
            A = torch.tensor(np.moveaxis(A_np.reshape(n,n,10), -1, 0)).reshape((10, -1)).to_sparse_coo()
        else:
            raise NotImplementedError
        d = A.size(0)
        assert d > n and top_k >= n
        # Always pick the first n power flow measurements
        M = torch.sparse_coo_tensor(
            torch.stack([torch.arange(n), torch.arange(n)], dim=0),
            torch.ones(n), 
            (d, d)
        )
        if not args.search_P:
            M = create_mask(
                logits2prob(torch.rand((d)), 'STE'),
                top_k, 
                'STE', 
                M
            )
        DSE_PSSE(
            top_k,
            args.search_loops,
            args.e_norm,
            A,
            f"{args.task}_{args.loss_type}",
            args.optimizer,
            args.iters,
            args.max_lr,
            args.min_lr,
            args.lr_sched,
            args.temperature,
            (args.max_sq_loss, args.min_sq_loss),
            (args.Vmin, )
        )


if __name__ == "__main__":
    main()