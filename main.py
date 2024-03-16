import argparse
from src.models.collaborative_filtering import base_collaborative_filtering
from src.models.collaborative_filtering_deep_learning import base_collaborative_filtering_deep_learning
from src.models.content_based_filtering import base_content_based_filtering
from src.models.helpers.user_genre_profile import UserGenreProfileGenerator
from src.models.content_based_filtering_deep_learning import base_content_based_filtering_deep_learning
from src.models.hybrid_filtering_deep_learning import base_hybrid_filtering_deep_learning

def main():
    parser = argparse.ArgumentParser(description='Filtering types for the project')
    parser.add_argument('--filter_cf', help='Type of collaborative filtering to perform (KNNBasic/NMF,DL)', required=False)
    parser.add_argument('--cbf_result', action='store_true', help='Enable checking of CBF results', required=False)
    parser.add_argument('--tune_dl', action='store_true', help='Enable hyperparamertuning mode for the DL model', required=False)
    parser.add_argument('--filter_cbf', help='Content based filtering if specified. Can be traditional or deep learning (TRAD, DL)', required=False)
    parser.add_argument('--init_user_profile', action='store_true', help='Initializes generation of user profiles for content based deep learning and prints them', required=False)
    parser.add_argument('--filter_hybrid', action='store_true', help='Hybrid deep learning based filtering', required=False)

    args = parser.parse_args()

    if (args.filter_cf is None and not args.filter_cbf and not args.init_user_profile and not args.filter_hybrid):
        print("Invalid arguments. For help use --help or -h")
        quit()
    
    if args.filter_hybrid:
        base_hybrid_filtering_deep_learning()
        quit()

    if args.filter_cbf == 'TRAD':
        base_content_based_filtering()
        if args.cbf_result:
            base_collaborative_filtering('KNNBasic', args.cbf_result)
        quit()
    elif args.filter_cbf == 'DL':
        base_content_based_filtering_deep_learning()
        quit()
    elif args.filter_cbf is not None:
        print('Invalid content based filtering mode. For help use --help or -h')
        quit()


    if args.init_user_profile:
        generator = UserGenreProfileGenerator()
        generator.build_profile(True)
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