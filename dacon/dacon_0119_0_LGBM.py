'''
LGBG
learning_rate을 줄이면서 n_estimator를 크게하는 것은 
부스팅에서의 기본적인 튜닝 방안

lightGBM에서 제공하는 파라미터
binary(Cross Entropy)
multiclass(Cross Entropy)
regression_l2(MSE)
regression_l1 (MAE)
mape (MAPE)
poisson (Log Transformation)
quantile (Quantile)
huber (Huber loss, MAE approx)
fair (Fair loss, MAE approx)
gamma (Residual Deviance)
lambdarank
tweedie


learning_rate : 일반적으로 0.01~ 0.1 정도로 맞추고 다른 파라미터를 튜닝한다
              : 나중에 성능을 더 높일 때 learning_rate를 더 줄인다.
num_iteration : 기본값이 100인데 1000정도는 해주는게 좋다. 너무 크게하면 
              : 과적합 발생할 수 있다.
max_depth : -1로 설정하면 제한없이 분기한다. feature가 많다면 크게 설정한다.
          : 파라미터 설정시 우선적으로 설정
boosting : 기본값은 gbdt이며 정확도가 중요할 때는 딥러닝의 드랍아웃과 같은
         : dart를 사용한다.
bagging_fraction. 배깅을 하기위해서 데이터를 랜덤 샘플링하여 학습에 사용한다. 
                  비율은 0 < fraction <= 1 이며 0이 되지 않게 해야한다.



            
'''

'''
import lightgbm as lgb
from lightgbm import LGBMRegressor
#LGBG
#learning_rate을 줄이면서 n_estimator를 크게하는 것은 
#부스팅에서의 기본적인 튜닝 방안
# Get the model and the predictions in (a) - (b)
def LGBM(q, x_train, y_train, x_valid, y_valid, x_test):
    
    # (a) Modeling  
    model = LGBMRegressor(objective='quantile', alpha=q,
                         n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)                   
                         
                         
    model.fit(x_train, y_train, eval_metric = ['quantile'], 
          eval_set=[(x_valid, y_valid)], early_stopping_rounds=300, verbose=500)
    # (b) Predictions
    pred = pd.Series(model.predict(x_test).round(2))
    return pred, model
'''