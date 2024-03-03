import argparse
from src.models.collaborative_filtering import base_collaborative_filtering
from src.models.collaborative_filtering_deep_learning import base_collaborative_filtering_deep_learning

def main():
    parser = argparse.ArgumentParser(description='Filtering types for the project')
    parser.add_argument('--filter_cf', help='Type of collaborative filtering to perform (KNNBasic/NMF,DL)', required=False)
    parser.add_argument('--cbf_result', action='store_true', help='Enable checking of CBF results', required=False)
    parser.add_argument('--filter_cbf', action='store_true', help='Content based filtering if specified', required=False)

    args = parser.parse_args()

    if (args.filter_cf == '' and not args.filter_cbf):
        print("Invalid arguments. For help use --help or -h")
        quit()

    if args.filter_cf == 'KNNBasic':
        base_collaborative_filtering('KNNBasic', args.cbf_result)
    elif args.filter_cf == 'NMF':
        base_collaborative_filtering('NMF', args.cbf_result)
    elif args.filter_cf == 'DL':
        base_collaborative_filtering_deep_learning()
    else:
        print('Invalid collaborative filtering mode. For help use --help or -h')
        quit()

if __name__ == "__main__":
    main()