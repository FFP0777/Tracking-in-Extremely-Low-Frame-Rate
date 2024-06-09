# 两种方法都能打开
import pickle
import numpy as np

f = open(r'D:\vehicle_mtmc\output\race_result\0902_130006_131041\mot.pkl','rb')
data = pickle.load(f)
print(data)

# img_path = './train_data.pkl'
# img_data = np.load(img_path)
# print(img_data)

