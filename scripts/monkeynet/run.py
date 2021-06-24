from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "transfer", "prediction"])
    parser.add_argument("--log_dir", default="../../monkeynet/log", help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default=0, type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")

    