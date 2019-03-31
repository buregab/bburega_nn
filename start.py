import sys 
from load_mnist import load_mnist
from network import Network

import numpy as np

def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    print("Load MNIST data")
    train, validation, test = load_mnist()

    print("Create network")
    network = Network(3, 3, 0.5, train[0:2])

    print("Run gradient descent")
    network.grad_descent()

    # Do argument parsing here (eg. with argparse) and anything else
    # you want your project to do.

if __name__ == "__main__":
    main()