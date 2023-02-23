# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 23:12:31 2022

@author: yuyue
"""

'''
GRU网格搜索                上海IMF1

2022年12月5日 19点10分
问题：
    1.lstm网络结构设置是否合理？
    2.超参数范围选择
    3.归一化和反归一化设置是否合理
    4.训练集、测试集划分是否合理
    5.将计算结果保存
    6.设置imf自动选择
'''

from math import sqrt

from numpy import array
from numpy import mean

from pandas import DataFrame
from pandas import concat
from pandas import read_excel

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout



# 划分训练集、测试集
def train_test_split(data, n_test):   
    return data[:-n_test], data[-n_test:]


# 转换为监督学习数据
def series_to_supervised(data, n_in=1, n_out=1):
	df = DataFrame(data)
	cols = list()
	# 输入序列 (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# 预测序列 (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# 组合在一起
	agg = concat(cols, axis=1)
	# 删除有NaN值的行
	agg.dropna(inplace=True)
	return agg.values


# rems
def measure_rmse(actual, predicted):
    
	return sqrt(mean_squared_error(actual, predicted))


# 构建lstm
def model_fit(train, config):
    # 读取超参数
    '''
    n_input 
    n_layer 
    n_nodes_input 
    n_nodes_1 
    n_nodes_2 
    n_nodes_3 
    n_epochs 
    n_batch     
    '''
    n_input, n_layer, n_nodes_input, n_nodes_1, n_nodes_2, n_nodes_3, n_epochs, n_batch = config
    # 转换数据
    data = series_to_supervised(train, n_in=n_input)
    # 划分数据
    train_x, train_y = data[:,:-1], data[:,-1]
    # 重塑数据
    n_features = 1
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], n_features))
    # 定义模型
    model = Sequential()
    
    # 1 GRU隐藏层    n_nodes_input, n_input, n_nodes_1,
    if n_layer == 1:
        model.add(Dense(n_nodes_input, activation='tanh', input_shape=(n_input, n_features)))
        model.add(GRU(n_nodes_1, return_sequences=False))
        model.add(Dense(1, activation= 'linear'))
        
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.001))
        model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
        
    # 2 GRU隐藏层   n_nodes_input, n_input, n_nodes_1, n_nodes_2,  n_dropout_rate, 
    # n_learn_rate, n_batch, n_epochs
    if n_layer == 2:
        model.add(Dense(n_nodes_input, activation='tanh', input_shape=(n_input, n_features)))
        model.add(GRU(n_nodes_1, return_sequences= True))
        model.add(Dropout(0.2))
        model.add(GRU(n_nodes_2, return_sequences=False))
        model.add(Dense(1, activation= 'linear'))
        
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.001))
        model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    
    # 3 GRU隐藏层  n_nodes_input, n_input, n_nodes_1, n_nodes_2,  n_nodes_3,
    # n_dropout_rate, n_learn_rate, n_batch, n_epochs
    if n_layer == 3:
        model.add(Dense(n_nodes_input, activation='tanh', input_shape=(n_input, n_features))) 
        model.add(GRU(n_nodes_1, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(n_nodes_2, return_sequences=True))
        model.add(GRU(n_nodes_3, return_sequences=False))
        model.add(Dense(1, activation= 'linear'))
        
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.001))
        model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)
    
    return model


# 拟合模型并预测
def model_predict(model, history, config):
    # 导入超参数n_input
    '''
    n_input 
    n_layer 
    n_nodes_input 
    n_nodes_1 
    n_nodes_2 
    n_nodes_3 
    n_epochs 
    n_batch   
    '''  
    n_input, _, _, _, _, _, _, _ = config
    x_input = array(history[-n_input:]).reshape((1, n_input, 1))
    yhat = model.predict(x_input, verbose=0)
    res = yhat[0]
    
    return res


# 前向验证得到评价指标
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # 分割数据集
    train, test = train_test_split(data, n_test)
    # 拟合模型
    model = model_fit(train, cfg)
    # 训练集添加到history
    history = [x for x in train]
    # 每次预测一步
    for i in range(len(test)):
        # 拟合模型，对history预测
        yhat = model_predict(model, history, cfg)
        # 将预测信息存储到predications中
        predictions.append(yhat)
        # 将实际值添加到history中，为下一次预测做准备
        history.append(test[i])
        
    # 逆归一化处理
    predictions = norm_scaler.inverse_transform(array(predictions))
    test = norm_scaler.inverse_transform(test)
    # 估算预测误差
    error = measure_rmse(test, predictions)
    print(' > %.3f' % error)
    return error
    

# 模型进行评估，失败时返回None；每组方案运行10次，取平均值
def repeat_evaluate(data, config, n_test, n_repeats=5):
    # 将配置转换为一个键
    key = str(config)
    # 拟合和评估模型n次
    scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
    # 总结得分
    result = mean(scores)
    print('> Model[%s] %.3f' % (key, result))
    return (key, result)


# 网格搜索配置
def grid_search(data, cfg_list, n_test):
	# 评估配置
	scores = [repeat_evaluate(data, cfg, n_test) for cfg in cfg_list]
	# 按误差对配置进行升序排序
	scores.sort(key=lambda tup: tup[1])
	return scores


# 创建配置列表
def model_configs():
    # 超参数范围
    # n_input = [8,12]
    # n_layer = [2,3]
    # n_nodes_input = [64,128]
    # n_nodes_1 = [128]
    # n_nodes_2 = [32,64]
    # n_nodes_3 = [32,64]
    # n_epochs = [100,300]
    # n_batch = [32,64]
    
    n_input = [8]
    n_layer = [2,3]
    n_nodes_input = [64,128]
    n_nodes_1 = [128]
    n_nodes_2 = [32]
    n_nodes_3 = [32]
    n_epochs = [100]
    n_batch = [32,64]

    configs = list()
    for i in n_input:
        for j in n_layer:
            for k in n_nodes_input:
                for l in n_nodes_1:
                    for m in n_nodes_2:
                        for n in n_nodes_3:
                            for o in n_epochs:
                                for p in n_batch:
                                    cfg = [i,j,k,l,m,n,o,p]
                                    configs.append(cfg)
    print('Total configs: %d' % len(configs))
    
    return configs
    

# ================================调用函数====================================
# 读取数据，并归一化处理
series = read_excel(r'D:\预测代码\1204\shanghaiGRU\上海港.xlsx')
choose_imf = 'IMF1'
norm_scaler = MinMaxScaler()
data = norm_scaler.fit_transform(series[[choose_imf]].values)
# 数据分割，前13年做训练集，后3年为测试集
n_test = 36
# 模型配置
cfg_list = model_configs()
# 网格搜索
scores = grid_search(data, cfg_list, n_test)
print('done')
# 排名前十的配置
for cfg, error in scores[:3]:
	print(cfg, error)


















