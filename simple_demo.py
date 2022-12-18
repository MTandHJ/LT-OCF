


import argparse

parser = argparse.ArgumentParser(description="Go LT-NCF")

parser.add_argument('--adjoint', type=bool, default=False, choices=[True, False])


args = parser.parse_args()


print(args.adjoint)