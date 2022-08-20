import numpy as np
import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty, ElementFraction, Meredig
import re
from tqdm import tqdm
import warnings


"""
功能: 定义一个获取元素特征的类
参数: 传入csv或者xlsx文件, 保证某一列是化学式(命名为Chemical_Formula)即可
结果: 
(1) 在原始文件中自动添加matminer中的特征列以及自定义函数的特征列
(2) 过滤重复意义的特征列信息, 并进行缺失值的填充
(3) 保存文件到本地指定路径
"""
class GetElementFeature(object):
    def __init__(self, file, col_name, element_property):
        self.file = file;
        self.col_name = col_name;
        self.element_property = element_property;

    # 主流程函数
    def mainPipe(self):
        df_cp = self.formulaToComposition(self.file, self.col_name);
        sources = ["magpie", "deml", "pymatgen", "matscholar_el", "megnet_el"];
        for source in sources:
            df 
        df_feat = self.compositionToProperty(df_cp, source = "magpie");
        df_feat = self.compositionToProperty(df_feat, source = "magpie");
        df_feat = self.compositionToProperty(df_feat, source = "magpie");
        df_feat = self.compositionToProperty(df_feat, source = "magpie");
        df_feat = self.addProperty(df_feat, self.element_property)

        return df

    # 字符串化学式转为"composition类"
    def formulaToComposition(self, file, col_name):
        stc = StrToComposition();
        try:
            df_cp = stc.featurize_dataframe(df = file, col_id = col_name, ignore_errors = False)
        except ValueError:
            print("Input formula with certain format error! ")
        print("None!")
        df_cp = df_cp.dropna(how = "any", axis=0);
        print("Shape of dataframe is {:}".format(df_cp.shape));

        return df_cp;

    # 从composition类提取特征
    def compositionToProperty(self, df, ):
        df = self.magpieProperty(df)
        df = self.demlProperty(df)
        df = self.matminerProperty(df)
        df = self.matscholarProperty(df)
        df = self.megnetProperty(df)


    # 人工增加特征函数
    def addProperty(self, df, element_property):
        pass


def loadData(file_path):
    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        data = pd.read_excel(file_path)
    elif file_path.endswith(".feather"):
        data = pd.read_feather(file_path)
    else:
        print("Input data format is wrong!")

    return data


def saveData(df, save_path, mode="csv"):
    if mode == "csv":
        df.to_csv(save_path, index = False)
    elif mode == "xlsx":
        df.to_excel(save_path, index=False)
    elif mode == "feather":
        df.to_feather(save_path, index=False)
    print("Success to save .csv file!")


if __name__ == "__main__":
    warnings.formatwarnings("ignore", category=FutureWarning)
    DATA_DIR = "D:\python\data\MAT"
    data_path = DATA_DIR + "\MD.xlsx"
    formula_col = "Chemical_Formula"
    eleprop_path = DATA_DIR + "\element_properties.csv"
    data = loadData(data_path)
    element_property = loadData(eleprop_path)
    GEF = GetElementFeature(data, formula_col, element_property)
    df_ok = GEF.mainPipe();
    save_path = DATA_DIR + "df_ok.csv"
    saveData(df_ok, save_path, mode="csv");

        
