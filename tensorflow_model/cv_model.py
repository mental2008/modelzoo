import tensorflow as tf
from tensorflow import keras

import os

models = {
    'DenseNet121': [None,224,224,3],
    'DenseNet169': [None,224,224,3],
    'DenseNet201': [None,224,224,3],
    'EfficientNetB0': [None,224,224,3],
    'EfficientNetB1': [None,240,240,3],
    'EfficientNetB2': [None,260,260,3],
    'EfficientNetB3': [None,300,300,3],
    'EfficientNetB4': [None,380,380,3],
    'EfficientNetB5': [None,456,456,3],
    'EfficientNetB6': [None,528,528,3],
    'EfficientNetB7': [None,600,600,3],
    'InceptionResNetV2': [None,299,299,3],
    'InceptionV3': [None,299,299,3],
    'MobileNet': [None,224,224,3],
    'MobileNetV2': [None,224,224,3],
    'NASNetLarge': [None,331,331,3],
    'NASNetMobile': [None,224,224,3],
    'ResNet101': [None,224,224,3],
    'ResNet152': [None,224,224,3],
    'ResNet50': [None,224,224,3],
    'ResNet101V2': [None,224,224,3],
    'ResNet152V2': [None,224,224,3],
    'ResNet50V2': [None,224,224,3],
    'VGG16': [None,224,224,3],
    'VGG19': [None,224,224,3],
    'Xception': [None,299,299,3]
}


if __name__ == '__main__':
    for model in models:
        pretrained_model = eval('keras.applications.{}()'.format(model))
        save_path = os.path.join(os.getcwd(), 'tensorflow_model', 'cv', model.lower(), '1')
        if os.path.exists(save_path):
            continue
        print(save_path)
        
        @tf.function()
        def my_predict(inputs):
            prediction = pretrained_model(inputs)
            return {"predictions": prediction}

        my_signatures = my_predict.get_concrete_function(
            inputs=tf.TensorSpec(shape=models[model], dtype=tf.dtypes.float32, name="inputs")
        )
        
        tf.saved_model.save(pretrained_model, save_path, signatures=my_signatures)
