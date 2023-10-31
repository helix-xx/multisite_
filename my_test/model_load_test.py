import os
from tqdm import tqdm
import tensorflow as tf
from moldesign.score.nfp import evaluate_mpnn, retrain_mpnn, NFPMessage, custom_objects
def task(*args,a=1,b=2,c=3):
    for arg in args:
        print(arg)
        
    # print(args[1])
    a,=args
    print(a)
    print(a,b,c)
    return a+b+c

if __name__ == '__main__':
    # print("main")
    # task(1,a=4,b=5,c=6)

    mpnn_file="/home/lizz_lab/cse12232433/project/colmena/multisite_/data/moldesign/initial-model/networks/"
    # get all path  
    for root,dir,files in os.walk(mpnn_file):
        print(root)
        
    for path in os.listdir(mpnn_file):
        path=os.path.join(mpnn_file,path)
        print(path)
        
    models = [
    tf.keras.models.load_model(os.path.join(mpnn_file,path,"model.h5") , custom_objects=custom_objects)
    for path in tqdm(os.listdir(mpnn_file), desc='Loading models')
    ]
    
    # # 获取所有模型文件路径
    # model_files = [os.path.join(mpnn_file, file) for file in os.listdir(mpnn_file)]

    # # 使用tqdm展示读取模型的过程
    # models = []
    # for model_file in tqdm(model_files, desc='Loading models'):
    #     model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
    #     models.append(model)