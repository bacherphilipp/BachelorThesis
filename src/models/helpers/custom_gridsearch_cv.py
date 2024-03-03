from itertools import product
import time
from joblib import Parallel, delayed
import numpy as np
from surprise import Dataset, Reader, SlopeOne
from surprise.dataset import DatasetUserFolds
from surprise.model_selection import GridSearchCV
from surprise.model_selection.search import get_cv
from surprise.model_selection.validation import accuracy

class CustomGridSearchCV(GridSearchCV):

    def filter_data(self, df, beyms=None, group=None):
        if beyms != None:
            df = df[df['isbyms'] == beyms]
        if group != None and beyms == 1:
            df = df[df['usergroup'] == group]
        
        return df[['user_id', 'item_id', 'rating']]
        

    def fit_and_score(self, algo, trainset, testset, measures, group_sets, return_train_measures=False):
        start_fit = time.time()
        algo.fit(trainset)
        fit_time = time.time() - start_fit
        start_test = time.time()
        predictions = algo.test(testset)
        test_time = time.time() - start_test

        if return_train_measures:
            train_predictions = algo.test(trainset.build_testset())

        test_measuresAll = dict()
        test_measuresMS = dict()
        test_measuresBYMS = dict()
        test_measuresUGR0 = dict()
        test_measuresUGR1 = dict()
        test_measuresUGR2 = dict()
        test_measuresUGR3 = dict()

        train_measures = dict()

        pred_ms = list()
        pred_byms = list()
        pred_UGR0 = list()
        pred_UGR1 = list()
        pred_UGR2 = list()
        pred_UGR3 = list()
        for pred in predictions:
            if pred.uid in group_sets['ms']:
                pred_ms.append(pred)
            if pred.uid in group_sets['byms']:
                pred_byms.append(pred)
            if pred.uid in group_sets['UGR0']:
                pred_UGR0.append(pred)
            if pred.uid in group_sets['UGR1']:
                pred_UGR1.append(pred)
            if pred.uid in group_sets['UGR2']:
                pred_UGR2.append(pred)
            if pred.uid in group_sets['UGR3']:
                pred_UGR3.append(pred)

        for m in measures:
            f = getattr(accuracy, m.lower())
            test_measuresAll[m] = f(predictions, verbose=0)            
            test_measuresMS[m] = f(pred_ms, verbose=0)
            test_measuresBYMS[m] = f(pred_byms, verbose=0)
            test_measuresUGR0[m] = f(pred_UGR0, verbose=0)
            test_measuresUGR1[m] = f(pred_UGR1, verbose=0)
            test_measuresUGR2[m] = f(pred_UGR2, verbose=0)
            test_measuresUGR3[m] = f(pred_UGR3, verbose=0)

        test_measures_list = [
                    test_measuresAll,
                    test_measuresMS,
                    test_measuresBYMS,
                    test_measuresUGR0,
                    test_measuresUGR1,
                    test_measuresUGR2,
                    test_measuresUGR3
        ]
        # MODIFIED also return predictions to filter various user groups
        return test_measures_list, train_measures, fit_time, test_time    
    
    def fit(self, data):
            raw_data = data
            reader = Reader(rating_scale=(1,1000))
            data = Dataset.load_from_df(self.filter_data(data), reader=reader)
            
            #userids of various groups
            groups = dict()
            groups['all'] = [-1]
            groups['ms'] = raw_data[raw_data["isbyms"] == 0]["user_id"].tolist()
            groups['byms'] = raw_data[raw_data["isbyms"] == 1]["user_id"].tolist() 
            groups['UGR0'] = raw_data[raw_data["usergroup"] == 0]["user_id"].tolist() 
            groups['UGR1'] = raw_data[raw_data["usergroup"] == 1]["user_id"].tolist() 
            groups['UGR2'] = raw_data[raw_data["usergroup"] == 2]["user_id"].tolist()
            groups['UGR3'] = raw_data[raw_data["usergroup"] == 3]["user_id"].tolist() 

            #for performance, convert group to set
            group_sets = {key: set(value) for key, value in groups.items()}

            if self.refit and isinstance(data, DatasetUserFolds):
                raise ValueError(
                    "refit cannot be used when data has been "
                    "loaded with load_from_folds()."
                )

            cv = get_cv(self.cv)           
            delayed_list = (
                delayed(self.fit_and_score)(
                    self.algo_class(**params) if not isinstance(self.algo_class(), SlopeOne) else self.algo_class(),
                    trainset,
                    testset,
                    self.measures,
                    group_sets,
                    self.return_train_measures
                )
                for params, (trainset, testset) in product(
                    self.param_combinations, cv.split(data)
                )
            )
            out = Parallel(
                n_jobs=self.n_jobs,
                pre_dispatch=self.pre_dispatch,
                verbose=self.joblib_verbose,
                timeout=99999
            )(delayed_list)
 

            #(test_measures_dicts_list, train_measures_dicts, fit_times, test_times) = zip(*out)

            fit_times = list()
            test_times = list()
            test_measures_dicts_list = list()
            for _ in range(7):
                test_measures_dicts_list.append(list())

            for tup in out:
                list_of_dicts = tup[0]
                fit_times.append(tup[2])
                test_times.append(tup[3])
                for i, d in enumerate(list_of_dicts):
                    test_measures_dicts_list[i].append(d)

            test_measures_list = list()
        
            #this is done for every group, so I get the optimal parameters for each group
            for test_measures_dicts in test_measures_dicts_list:
                test_measures = dict()
                train_measures = dict()
                new_shape = (len(self.param_combinations), cv.get_n_folds())
                for m in self.measures:
                    test_measures[m] = np.asarray([d[m] for d in test_measures_dicts])
                    test_measures[m] = test_measures[m].reshape(new_shape)
                test_measures_list.append(test_measures)

            gridsearch_resultlist = list()

            for test_measures, group in zip(test_measures_list,['all', 'ms', 'byms', 'UGR0','UGR1','UGR2','UGR3']):
                cv_results = dict()
                best_index = dict()
                best_params = dict()
                best_score = dict()
                best_estimator = dict()
                for m in self.measures:
                    # cv_results: set measures for each split and each param comb
                    for split in range(cv.get_n_folds()):
                        cv_results[f"split{split}_test_{m}"] = test_measures[m][:, split]
                        if self.return_train_measures:
                            cv_results[f"split{split}_train_{m}"] = train_measures[m][:, split]

                    # cv_results: set mean and std over all splits (testset and
                    # trainset) for each param comb
                    mean_test_measures = test_measures[m].mean(axis=1)
                    cv_results[f"mean_test_{m}"] = mean_test_measures
                    cv_results[f"std_test_{m}"] = test_measures[m].std(axis=1)

                    # cv_results: set rank of each param comb
                    # also set best_index, and best_xxxx attributes
                    indices = cv_results[f"mean_test_{m}"].argsort()
                    cv_results[f"rank_test_{m}"] = np.empty_like(indices)
                    if m in ("mae", "rmse", "mse"):
                        cv_results[f"rank_test_{m}"][indices] = (
                            np.arange(len(indices)) + 1
                        )  # sklearn starts at 1 as well
                        best_index[m] = mean_test_measures.argmin()
                    elif m in ("fcp",):
                        cv_results[f"rank_test_{m}"][indices] = np.arange(len(indices), 0, -1)
                        best_index[m] = mean_test_measures.argmax()
                    best_params[m] = self.param_combinations[best_index[m]]
                    best_score[m] = mean_test_measures[best_index[m]]
                    #best_estimator[m] = self.algo_class(**best_params[m]) if not isinstance(self.algo_class(), SlopeOne) else self.algo_class(),
                    best_estimator[m] = self.algo_class(**best_params[m])

                # Cv results: set fit and train times (mean, std)
                fit_times = np.array(fit_times).reshape(new_shape)
                test_times = np.array(test_times).reshape(new_shape)
                for s, times in zip(("fit", "test"), (fit_times, test_times)):
                    cv_results[f"mean_{s}_time"] = times.mean(axis=1)
                    cv_results[f"std_{s}_time"] = times.std(axis=1)

                # cv_results: set params key and each param_* values
                cv_results["params"] = self.param_combinations
                for param in self.param_combinations[0]:
                    cv_results["param_" + param] = [
                        comb[param] for comb in self.param_combinations
                    ]

                if self.refit:
                    best_estimator[self.refit].fit(data.build_full_trainset())

                self.best_estimator = best_estimator

                GridSearchResult(group,cv_results,best_index,best_params, best_score,best_estimator)
                gridsearch_resultlist.append(GridSearchResult(group,cv_results,best_index,best_params, best_score,best_estimator))

            self.gridsearch_results = gridsearch_resultlist

class GridSearchResult:
    def __init__(self, group, cv_results, best_index, best_params, best_score, best_estimator):
        self
        self.group = group
        self.cv_results = cv_results
        self.best_index = best_index
        self.best_params = best_params
        self.best_score = best_score
        self.best_estimator = best_estimator
