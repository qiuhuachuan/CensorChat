import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_file', type=str, required=True)
    parser.add_argument('--cuda_no', type=str, default='0')

    args = parser.parse_args()
    return args