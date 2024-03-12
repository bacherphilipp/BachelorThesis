import argparse
from src.models.collaborative_filtering import base_collaborative_filtering
from src.models.collaborative_filtering_deep_learning import base_collaborative_filtering_deep_learning
from src.models.content_based_filtering import base_content_based_filtering
from src.models.helpers.user_genre_profile import base_build_user_profile

def main():
    parser = argparse.ArgumentParser(description='Filtering types for the project')
    parser.add_argument('--filter_cf', help='Type of collaborative filtering to perform (KNNBasic/NMF,DL)', required=False)
    parser.add_argument('--cbf_result', action='store_true', help='Enable checking of CBF results', required=False)
    parser.add_argument('--tune_dl', action='store_true', help='Enable hyperparamertuning mode for the DL model', required=False)
    parser.add_argument('--filter_cbf', action='store_true', help='Content based filtering if specified', required=False)
    parser.add_argument('--init_user_profile', action='store_true', help='Initializes generation of user profiles for content based deep learning', required=False)

    args = parser.parse_args()

    if (args.filter_cf is None and not args.filter_cbf and not args.init_user_profile):
        print("Invalid arguments. For help use --help or -h")
        quit()
    
    if args.filter_cbf:
        base_content_based_filtering()
        if args.cbf_result:
            base_collaborative_filtering('KNNBasic', args.cbf_result)
        quit()

    if args.init_user_profile:
        base_build_user_profile()
        quit()

    if args.filter_cf == 'KNNBasic':
        base_collaborative_filtering('KNNBasic', args.cbf_result)
    elif args.filter_cf == 'NMF':
        base_collaborative_filtering('NMF', args.cbf_result)
    elif args.filter_cf == 'DL':
        base_collaborative_filtering_deep_learning(args.tune_dl)
    else:
        print('Invalid collaborative filtering mode. For help use --help or -h')
        quit()

if __name__ == "__main__":
    main()