#저장된 파일 불러오기
from tensorflow.keras.models import load_model
model = load_model("../data/h5/test_save_keras35_1.h5")

model.summary()