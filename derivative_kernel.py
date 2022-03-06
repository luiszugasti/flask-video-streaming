import numpy as np
import sys

'''
================================================================================
derivative_kernel.py
usage: python3 derivative_kernel.py H
    H: desired horizontal kernel, from four choices. cd, fd, p or s.
Provides 4 different kinds of horizontal kernel matrices to the user's calling 
function.
Depending on the user's input after the call to this file, a kernel will be
returned.
        cd: Central Difference             [1 0 -1]
        fd: Forward Difference             [0 1 -1]
        p: Prewitt                         [1 0 -1]
                                           [1 0 -1]
                                           [1 0 -1]
        s: Sobel                           [1 0 -1]
                                           [2 0 -2]
                                           [1 0 -1]
If an invalid input is detected, the program will return [1] (no derivative) 
and will send a message to the command line.
To invoke the help docs, use python3 derivative_kernel.py --help
================================================================================
'''


def derivative_kernel_return(id):
    if id == "cd":
        return np.array([[1, 0, -1]], np.float64)
    elif id == "fd":
        return np.array([[0, 1, -1]], np.float64)
    elif id == "p":
        return np.array([[1, 0, -1], \
                         [1, 0, -1], \
                         [1, 0, -1]], np.float64)
    elif id == "s":
        return np.array([[1, 0, -1], \
                         [2, 0, -2], \
                         [1, 0, -1]], np.float64)
    else:
        print("invalid value for id")
        return np.array([1], np.float64)  # error


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "--help":
            print("\n\nusage: python3 derivative_kernel.py H\n\
    H: desired horizontal kernel, from four choices. cd, fd, p or s.\n\
    Provides 4 different kinds of horizontal kernel matrices to the user's\n\
    calling function.\n\
    Depending on the user's input after the call to this file, a kernel\n\
    will be returned.\n\
        cd: Central Difference             [1 0 -1]\n\n\
        fd: Forward Difference             [0 1 -1]\n\n\
        p: Prewitt                         [1 0 -1]\n\
                                           [1 0 -1]\n\
                                           [1 0 -1]\n\n\
        s: Sobel                           [1 0 -1]\n\
                                           [2 0 -2]\n\
                                           [1 0 -1]\n\n\
    If an invalid input is detected, the program will return [1]\n\
    (no derivative) and will send a message to the command line.\n\
    To invoke the help docs, use python3 derivative_kernel.py --help\n")
            exit()
        else:
            print(derivative_kernel_return(sys.argv[1].lower()))
            exit()
    else:
        print("\n\nusage: python3 derivative_kernel.py H\n\
            H: filtering kernel in horizontal. Replace H with --help\n\
            parameter to find out about optional input parameters")
    exit()
