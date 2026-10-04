"""Train v30 (no shared ViT; RGB 24 + isolated 8-layer depth prep)."""

from train_loop import main
import argparse
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train v30 (superpoint encoder + split RGB/depth grid decoder)"
    )
    parser.add_argument("--py-config", default="config.py")
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument("--iter-resume", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    if args.gpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
