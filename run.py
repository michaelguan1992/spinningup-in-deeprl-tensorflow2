import sys


def main(test_algo=None):

  if test_algo == 'vpg' or sys.argv[1] == 'vpg':
    from algos import vpg
    vpg.main()


if __name__ == '__main__':
  main(test_algo='vpg')
