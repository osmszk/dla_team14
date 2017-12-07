# python 2.x
#doc https://apple.github.io/coremltools/generated/coremltools.converters.keras.convert.html

path = '../keras/model_member.h5'

import coremltools
# import keras
# model = keras.models.load_model(path)
coreml_model = coremltools.converters.keras.convert(path,
        input_names = 'image',
        image_input_names = 'image',
        is_bgr = True,
        image_scale = 0.00392156863,
        class_labels = 'labels.txt')

coreml_model.save('MemberPredict.mlmodel')
