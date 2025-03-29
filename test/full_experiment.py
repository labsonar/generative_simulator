import subprocess
from argparse import ArgumentParser

def main(args) -> None:
    classes = ['A','B','C','D']
    for classe in classes:
        subprocess.run(["python","test/model_training.py", "--max_steps", args.max_steps, f"./model/{classe}", f"/data/4classes_15s/{classe}"], check=True)
        subprocess.run(["python","test/model_inference.py", "-t", args.n_samples, "-o", f"./audio/{classe}", f"./model/{classe}/weights.pt"], check = True)
    subprocess.run(["python","test/spectral_plot.py","./audio/A", './audio/B', './audio/C', './audio/D', '/data/4classes_15s/A', '/data/4classes_15s/B', '/data/4classes_15s/C', '/data/4classes_15s/D'])

if __name__ == "__main__":
    parser = ArgumentParser(description='realize full experiment (train,inference and t-sne)')
    parser.add_argument('--max_steps', '-m', default=None, type=str,
    help='maximum number of training steps')
    parser.add_argument('--n_samples', '-s', default=None, type=str ,
    help='number of samples')
    main(parser.parse_args())

