import os
import pandas as pd
import csv

RESULT_HOME_DIR = "results/"
METRIC_FILE = "metric.csv"

_data = "data"
_key = "image"
_m_psnr = "psnr"  # up
_m_mse = "mse"  # down
_m_ssim = "ssim"  # up


def isdir(f):
    return os.path.isdir(f)


def c(d, f):
    return os.path.join(d, f)


def is_result_dir(d):
    mf = os.path.join(d, METRIC_FILE)
    return os.path.exists(mf)


def yield_expirements_dir():
    for f in os.listdir(RESULT_HOME_DIR):
        d = c(RESULT_HOME_DIR, f)
        if isdir(d):
            if is_result_dir(d):
                yield d
            for f in os.listdir(d):
                sd = c(d, f)
                if isdir(sd) and is_result_dir(sd):
                    yield sd


def has_header(f):
    for l in f.readlines():
        return l.startswith("image")
    return False


def parse_expierement(d):

    f = os.path.join(d, METRIC_FILE)
    with open(f, "r") as _f:

        has_header = False
        for l in _f.readlines():
            has_header = l.startswith("image")
            break
        _f.seek(0)
        if has_header:
            df = pd.read_csv(_f)
            df = df.drop(columns=["image"])
            print(df.mean())
        else:
            rd = csv.DictReader(_f, [_key, _m_psnr, _m_mse, _m_ssim])
            er = {}
            for r in rd:
                label = r[_key]
                r.pop(_key)
                er[label] = {k: float(v) for k, v in r.items()}
            df = pd.DataFrame.from_dict(er, orient="index")
            print(df.mean())


def test_level_metric(er):
    # df = pd.DataFrame.from_dict(er, orient="index")
    # print(df.mean())
    pass


def parse():
    for d in yield_expirements_dir():
        print("result of ", d)
        parse_expierement(d)
        print("\n\n")


if __name__ == "__main__":
    parse()
