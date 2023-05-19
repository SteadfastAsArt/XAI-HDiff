# @Author  : Andrian Lee
# @Time    : 2022/4/5 16:43
# @File    : models.py
import numpy as np
import pandas as pd
from utils import *
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

from collections import Counter

import shap

import matplotlib.pyplot as plt

import logging

logging.basicConfig(filename='tmp.log',
                    filemode='w',
                    format='[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s]%(message)s',
                    # datefmt='%H:%M:%S',
                    level=logging.INFO)


class DataManager:
    def __init__(self, data_path, sheet_name=0):
        """_:protected, __:private"""
        logging.info('%s read file start' % self.__class__.__name__)
        self._data_all = read_file(data_path, sheet_name=sheet_name)
        logging.info('%s read file end' % self.__class__.__name__)
        print(self._data_all.info())
        logging.info('{} Stats for {}:'.format(self.__class__.__name__, data_path))
        logging.info(self._data_all.info())
        self._X_col = None  # X-column names
        self._y_col = None  # y-column name
        self._data_X = None  #
        self._data_y = None  #
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None

    @property
    def data_all(self) -> pd.DataFrame:
        return self._data_all

    @data_all.setter
    def data_all(self, table_path: str):
        self._data_all = read_file(table_path)
        print('Stats for {}:'.format(table_path))
        print(self._data_all.info())

    @property
    def data_X(self) -> pd.DataFrame:
        return self._data_X

    @property
    def data_Y(self) -> pd.DataFrame:
        return self._data_y

    @property
    def X_train(self) -> pd.DataFrame:
        return self._X_train

    @property
    def X_test(self) -> pd.DataFrame:
        return self._X_test

    @property
    def y_train(self) -> pd.DataFrame:
        return self._y_train

    @property
    def y_test(self) -> pd.DataFrame:
        return self._y_test

    @property
    def X_col(self):
        return self._X_col

    def choose_X_y(self, idx_list_X, idx_y=None):
        self._X_col = self._data_all.columns[idx_list_X]  # TODO
        data_X = self._data_all[self._X_col]
        self._data_X = data_X

        if idx_y is not None:
            self._y_col = self._data_all.columns[idx_y]
            data_y = self._data_all[self._y_col]
            self._data_y = data_y
            print('Stats for data_y:', Counter(data_y))
            logging.info('{} Stats for data_y: {}'.format(self.__class__.__name__, Counter(data_y)))

    def impute(self, neighbor_cnt):
        logging.info('{} Imputation start'.format(self.__class__.__name__))
        imputer = KNNImputer(n_neighbors=neighbor_cnt)
        np_imp = imputer.fit_transform(self._data_X)
        data_X_imp = pd.DataFrame(data=np_imp, columns=self._X_col)
        self._data_X = data_X_imp
        logging.info('{} Imputation results: {}'.format(self.__class__.__name__, self._data_X.info()))
        print(self._data_X.info())

    def normalize(self, minmax: tuple = None):
        if minmax is None:
            x = self.data_X
            # y = self.data_Y
            self._data_X = (x - x.min()) / (x.max() - x.min())
            # self._data_y = (y-y.min()) / (y.max() - y.min())
            return x.min(), x.max()
        else:
            _min = minmax[0]
            _max = minmax[1]
            x = self.data_X
            self._data_X = (x - _min) / (_max - _min)

    def split_train_test(self, ratio):
        """
        Init the train & test for supervised training phase.
        :param ratio: =test/all
        :return:
        """
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._data_X, self._data_y,
                                                                                    test_size=ratio,
                                                                                    random_state=SEED)
        print('Training X:', self._X_train.shape, 'Testing X:', self._X_test.shape)
        print('Training Y:', self._y_train.shape, 'Testing y:', self._y_test.shape)
        logging.info('Training X:', self._X_train.shape, 'Testing X:', self._X_test.shape)
        logging.info('Training Y:', self._y_train.shape, 'Testing y:', self._y_test.shape)


class ModelBuilder:
    def __init__(self, fix_param: dict = None, option: str = 'xgb', model=None):
        self._model = model
        self._fix_param = fix_param
        self._option = option

    def initialize(self, ):
        if self._option == 'xgb':
            self._model = xgb.XGBClassifier(**self._fix_param, use_label_encoder=False, eval_metric='logloss',
                                            random_state=SEED, )
        elif self._option == 'svm':
            from sklearn.svm import SVC
            from sklearn.ensemble import RandomForestClassifier
            self._model = SVC()  # C=10, gamma=0.04, class_weight='balanced', random_state=SEED
            # self._model = RandomForestClassifier(random_state=SEED)

    @property
    def model(self):
        return self._model

    def train(self, data_mana: DataManager, usedata='train'):
        print('{} model training...'.format(self._option))
        if usedata == 'all':
            self._model.fit(data_mana.data_X, data_mana.data_Y)
            y_pred = self._model.predict(data_mana.data_X)
            classification_metrics(data_mana.data_Y, y_pred)
        elif usedata == 'train':
            self._model.fit(data_mana.X_train, data_mana.y_train)
            y_train_pred = self._model.predict(data_mana.X_train)
            classification_metrics(data_mana.y_train, y_train_pred)

    def test(self, data_mana: DataManager):
        print('{} model testing...'.format(self._option))
        y_test_pred = self._model.predict(data_mana.X_test)
        classification_metrics(data_mana.y_test, y_test_pred)

    def predict(self, data_mana: DataManager, new_col):
        print('{} model predicting...'.format(self._option))
        data_y_pred = self._model.predict(data_mana.data_X)
        # save prediction output to original data
        data_mana.data_all[new_col] = data_y_pred
        return data_y_pred

    def plot_tree(self, base_path, num_trees: int = 5, rankdir: str = 'UT'):  # 'LR'
        judge_path(base_path)
        for i in range(num_trees):
            # ax = xgb.plot_tree(self._model, num_trees=num_trees, rankdir=rankdir)
            xgb.to_graphviz(self._model, num_trees=i, rankdir=rankdir).render(base_path + 'tree_{}'.format(i),
                                                                              format='png')
        xgb.to_graphviz(self._model, num_trees=100, rankdir=rankdir).render(base_path + 'tree_{}'.format(100),
                                                                            format='png')

    def plot_confusion_matrix(self, base_path, data_mana: DataManager):
        judge_path(base_path)
        disp = ConfusionMatrixDisplay.from_estimator(self._model,
                                                     data_mana.X_test, data_mana.y_test, cmap=plt.cm.Blues,
                                                     display_labels={'Diffusion': 0, 'non-Diffusion': 1},
                                                     #       normalize='true',
                                                     )
        disp.ax_.set_title('confusion matrix')
        plt.savefig(base_path + 'confusionMat_{}.png'.format(self._option), bbox_inches='tight')
        plt.close()

    def plot_importance(self, base_path):
        judge_path(base_path)
        importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
        for f in importance_types:
            imp = self._model.get_booster().get_score(importance_type=f)
            xgb_imp_list = [(i, imp[i]) for i in imp]
            xgb_imp_list_sorted = sorted(xgb_imp_list, key=lambda x: x[1], reverse=False)
            print(xgb_imp_list_sorted)
            _x = [i[0] for i in xgb_imp_list_sorted]
            _y = [i[1] for i in xgb_imp_list_sorted]
            fig, ax = plt.subplots()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.barh(_x, _y)
            for i, v in enumerate(_y):
                plt.text(v, i - 0.1, " " + str(round(v, 1)), color='blue')
            plt.title('Feature Importance - ' + f)
            plt.savefig(base_path + 'feat_imp_{}.png'.format(f))
            plt.clf()  # TODO

    def plot_PDP_ICE(self, base_path, data_df: pd.DataFrame):
        judge_path(base_path)
        # from sklearn.inspection import partial_dependence
        from sklearn.inspection import plot_partial_dependence
        fig, ax = plt.subplots(figsize=(9, 12))
        # plt.rcParams['figure.dpi'] = 300

        features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        display = plot_partial_dependence(
            self._model, data_df, features=data_df.columns, response_method='auto', kind="both",
            subsample=200, ax=ax, grid_resolution=30, random_state=42,
        )
        display.figure_.subplots_adjust(wspace=0.1, hspace=0.3)
        print(display.axes_)
        display.axes_[0][0].get_legend().remove()
        # handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper center')
        plt.savefig(base_path + 'PDP_ICE_test.png', format='png')  # eps, pdf, svg
        plt.close()


class SHAPExplainer:
    """
    SHAP: model-agnostic
        + model to be explained
        + data to run through
    TreeSHAP:
        feature_perturbation='interventional':
            background dataset: required
        feature_perturbation='tree_path_dependent':
            background dataset: NOT required
    """

    def __init__(self, model_builder: ModelBuilder, data_mana: DataManager, base_path, bg_data: pd.DataFrame = None):
        """
        Wrapper Only for TreeSHAP now.

        :param model_builder: Trained model to be explained
        :param data: data to run through the model
        """
        self._model_builder = model_builder
        self._model = self._model_builder.model
        self._data_mana = data_mana

        self._explainer = shap.TreeExplainer(self._model, data=bg_data,
                                             feature_perturbation='tree_path_dependent' if bg_data is None else 'interventional', )
        self._shap_value = None
        self._shap_interaction = None

        self._base_path = base_path
        judge_path(self._base_path)

    def shap_value(self, out_file=None):
        # pass all the X data from DataManager
        shap_values = self._explainer(self._data_mana.data_X)
        self._shap_value = shap_values
        # output to disk
        df_shap = pd.DataFrame(shap_values.values, columns=['shap_' + i for i in self._data_mana.X_col])
        df_shap['base_value'] = shap_values.base_values
        data_all_shap_df = pd.concat([self._data_mana.data_all, df_shap], axis=1, )
        if out_file is not None:
            data_all_shap_df.out(self._base_path + out_file)

    def shap_interaction_value(self, out_file, a=None, b=None, ):
        judge_path(self._base_path + 'inter/')
        shap_interaction_values = self._explainer.shap_interaction_values(self._data_mana.data_X)
        print(shap_interaction_values.shape)
        # print('SHAP interaction value', np.sum(shap_interaction_values[0], axis=1))
        # print(self._shap_value.values[0])
        self._shap_interaction = shap_interaction_values

        inter_tmp = []
        for a_ind in range(len(self._data_mana.X_col)):
            for b_ind in range(a_ind, len(self._data_mana.X_col)):
                print(a_ind, b_ind)
                inter_ab = np.array([i[a_ind, b_ind] for i in shap_interaction_values])  # Do NOT *2 for now
                df_inter_ab = pd.Series(inter_ab,
                                        name=self._data_mana.X_col[a_ind] + '_' + self._data_mana.X_col[b_ind])
                inter_tmp.append(df_inter_ab)
        data_all_inter_ab = pd.concat([self._data_mana.data_all, pd.concat(inter_tmp, axis=1)], axis=1, )
        data_all_inter_ab.out(self._base_path + 'inter/' + out_file)

    def plot_summary(self):
        shap.summary_plot(self._shap_value, self._data_mana.data_X, color_bar_label=True, show=False, plot_type="bar")
        plt.savefig(self._base_path + 'shap_all_summary_bar.png', bbox_inches='tight')
        plt.clf()

        shap.summary_plot(self._shap_value, self._data_mana.data_X, color_bar_label=True, show=False, )
        # matplotlib 3.5 issue
        # plt.gcf().axes[-1].set_aspect(100)
        # plt.gcf().axes[-1].set_box_aspect(100)
        plt.savefig(self._base_path + 'shap_all_summary.pdf', bbox_inches='tight', format='pdf')  # eps, pdf, svg
        plt.close()

    def plot_shap_value(self):
        # for i in self._data_mana.X_col:
        #     # TODO: hardcode here
        #     shap.plots.scatter(self._shap_value[:, i], color=self._shap_value[:, RENAME_MAP["H2O"]],
        #                        show=False)  # show=False | shap_values[:, "H2O"], shap_values
        #     # plt.gcf().axes[-1].set_aspect(2)
        #     # plt.gcf().axes[-1].set_box_aspect(1)
        #     plt.savefig(self._base_path + 'SHAPval_' + i + '_H2O.png', bbox_inches='tight')
        #     plt.close()
        # for i in self._data_mana.X_col:
        #     shap.plots.scatter(self._shap_value[:, i], show=False, color=self._shap_value)
        #     plt.savefig(self._base_path + 'SHAPval_' + i + '.png', bbox_inches='tight')
        #     plt.close()
        for i in self._data_mana.X_col:
            shap.plots.scatter(self._shap_value[:, RENAME_MAP["H2O"]], show=False, color=self._shap_value[:, i])
            plt.savefig(self._base_path + 'SHAPval_H2O_' + i + '.png', bbox_inches='tight')
            plt.close()

    def plot_interaction(self):
        for i in self._data_mana.X_col:
            for j in self._data_mana.X_col:
                shap.dependence_plot(
                    (i, j),  # "H2O"
                    self._shap_interaction, self._data_mana.data_X, show=False,  # train_data_X, X_train
                )
                # plt.gcf().axes[-1].set_aspect(100)
                # plt.gcf().axes[-1].set_box_aspect(100)
                plt.savefig(self._base_path + 'inter/SHAPinter_' + i + '_' + j + '.png', bbox_inches='tight')
                # plt.clf()
                plt.close()

    def plot_waterfall(self, id_list):
        judge_path(self._base_path + 'waterfall/')
        for i in id_list:
            shap.plots.waterfall(self._shap_value[i], show=False)
            plt.savefig(self._base_path + 'waterfall/SHAPtrain_{}.png'.format(i), bbox_inches='tight')
            plt.close()
    
    def plot_heatmap(self):
        shap.plots.heatmap(self._shap_value, show=False)
        plt.savefig(self._base_path + 'heatmap.pdf', bbox_inches='tight', format='pdf')
        plt.close()


class ModelScheduler:
    pass
