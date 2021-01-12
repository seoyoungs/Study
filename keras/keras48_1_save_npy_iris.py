from sklearn.datasets import load_iris
import numpy as np

dataset = load_iris()
#print(dataset)
print(dataset.keys())
#dict_keys(['data', 'target', 'frame', 'target_names', 
#           'DESCR', 'feature_names', 'filename'])
'''
numpy 일때
x_data = dataset.data #플로트 형식
y_data = dataset.target #init형식
'''
#딕셔너리 버전
x_data=dataset['data']
y_data=dataset['target']
print(x_data)
print(y_data)
# [7.2 3.2 6.  1.8]이런 value가 key값에 있다.
print(dataset.frame)
print(dataset.target_names) #['setosa' 'versicolor' 'virginica']
print(dataset['DESCR'])
print(dataset['feature_names'])
print(dataset.filename)

print(type(x_data), type(y_data)) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

# numpy로 저장
np.save('../data/npy/iris_x.npy', arr=x_data)
np.save('../data/npy/iris_y.npy', arr=y_data)
