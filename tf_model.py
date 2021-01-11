import tensorflow as tf
from tensorflow import keras

import os

models = [
    # 'DenseNet121',
    # 'DenseNet169',
    # 'DenseNet201',
    # 'EfficientNetB0',
    # 'EfficientNetB1',
    # 'EfficientNetB2',
    # 'EfficientNetB3',
    # 'EfficientNetB4',
    # 'EfficientNetB5',
    # 'EfficientNetB6',
    # 'EfficientNetB7',
    # 'InceptionResNetV2',
    # 'InceptionV3',
    # 'MobileNet',
    # 'MobileNetV2',
    # 'NASNetLarge',
    # 'NASNetMobile',
    # 'ResNet101',
    # 'ResNet152',
    'ResNet50',
    # 'ResNet101V2',
    # 'ResNet152V2',
    # 'ResNet50V2',
    # 'VGG16',
    # 'VGG19',
    # 'Xception'
]


if __name__ == '__main__':
    for model in models:
        pretrained_model = eval('keras.applications.{}()'.format(model))
        save_path = os.path.join(os.getcwd(), 'tensorflow_model', model.lower(), '1')
        print(save_path)
        
        @tf.function()
        def my_predict(inputs):
            prediction = pretrained_model(inputs)
            return {"predictions": prediction}
        
        my_signatures = my_predict.get_concrete_function(
            inputs=tf.TensorSpec([None,224,224,3], dtype=tf.dtypes.float32, name="inputs")
        )
        
        tf.saved_model.save(pretrained_model, save_path, signatures=my_signatures)
