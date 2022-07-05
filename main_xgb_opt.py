# @Author  : Andrian Lee
# @Time    : 2022/4/15 15:55
# @File    : main_xgb_opt.py
from models import DataManager, ModelBuilder, SHAPExplainer
import xgboost as xgb
from utils import *
import numpy as np

plt.rcParams['savefig.dpi'] = 600  # 图片像素
plt.rcParams['figure.dpi'] = 600  # 分辨率

SEED = 42
# np.random.seed(SEED)
base_path = './model_test/'

dMana = DataManager(r"D:\pre_grad_research\MantleWater\HDiff-XAI\all_data_20220606.xlsx", sheet_name='train')
dMana.choose_X_y([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 20)
dMana.impute(neighbor_cnt=30)
dMana.split_train_test(ratio=0.2)

param_xgb_final = dict([('learning_rate', 0.15), ('n_estimators', 250),
                        ('max_depth', 6), ('min_child_weight', 1.1), ('gamma', 0.2),
                        ('subsample', 0.9), ('colsample_bytree', 1.0), ('reg_alpha', 0.001), ('reg_lambda', 1)])
model_xgb = ModelBuilder(fix_param=param_xgb_final)
model_xgb.initialize()
model_svm = ModelBuilder(option='svm')
model_svm.initialize()

"""Training"""
## XGBoost - non-norm
# model_xgb.train(dMana, usedata='all')
model_xgb.train(dMana)
model_xgb.test(dMana)
# model_xgb.plot_tree(base_path, 5, 'LR')
# model_xgb.plot_confusion_matrix(base_path, dMana)
# model_xgb.plot_PDP_ICE(base_path, dMana.X_train)
# model_xgb.plot_importance(base_path)

## SVM - norm
# dMana_norm = DataManager(r"D:\pre_grad_research\MantleWater\HDiff-XAI\all_data_20220606.xlsx", sheet_name='train')
# dMana_norm.choose_X_y([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 20)
# dMana_norm.impute(neighbor_cnt=30)
# xmin, xmax = dMana_norm.normalize()
# dMana_norm.split_train_test(ratio=0.2)
# model_svm.train(dMana_norm)
# model_svm.test(dMana_norm)
# model_svm.plot_confusion_matrix('./model/', dMana_norm)

"""Inference"""
# XGBoost - non-norm
dMana_infer = DataManager(r"D:\pre_grad_research\MantleWater\HDiff-XAI\all_data_20220606.xlsx", sheet_name='app')
dMana_infer.choose_X_y([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
# model_xgb.predict(dMana_infer, 'XGB_label')
# dMana_infer.data_all.out(base_path+'pred.xlsx')

## SVM - norm
# dMana_infer_norm = DataManager(r"D:\pre_grad_research\MantleWater\HDiff-XAI\all_data_20220606.xlsx", sheet_name='app')
# dMana_infer_norm.choose_X_y([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
# dMana_infer_norm.normalize((xmin, xmax))
# model_svm.predict(dMana_infer_norm, 'SVM_label')
# dMana_infer_norm.data_all.out('./model/pred_1.xlsx')

"""SHAP train"""
# shap = SHAPExplainer(model_xgb, dMana, base_path)
# shap.shap_value('xgb_shap.xlsx')
# shap.plot_summary()
# shap.plot_shap_value()
# shap.shap_interaction_value('xgb_shap_inter.xlsx')
# shap.plot_interaction()

"""SHAP train"""
shap_infer = SHAPExplainer(model_xgb, dMana_infer, base_path)
shap_infer.shap_value()
shap_infer.plot_waterfall(np.arange(13, 24))
