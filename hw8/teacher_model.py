# teacher model for knowledge distillation

import numpy as np

import hw3_models.model1
import hw3_models.model2
import hw3_models.model3
import hw3_models.model4
import hw3_models.model5
import hw3_models.model6
import hw3_models.model7
import hw3_models.model8

# ====================================

def normalize_1(x_data):
    return x_data / 255

def normalize_2(x_data, mean, std):
    return (x_data - mean) / std

# ====================================

class Teacher:
    '''
    A generator using the models in hw3 to predict images.

    # Attributes:
        models(list of keras model)
        generator(generator)
        batch_size(int)
        mean(float)
        std(float)
    '''
    def __init__(self, generator, batch_size, mean, std):
        models = list()
        model1 = hw3_models.model1.build_model()
        model1.load_weights('./hw3_models/model1.h5')
        models.append(model1)

        model2 = hw3_models.model2.build_model()
        model2.load_weights('./hw3_models/model2.h5')
        models.append(model2)

        model3 = hw3_models.model3.build_model()
        model3.load_weights('./hw3_models/model3.h5')
        models.append(model3)

        model4 = hw3_models.model4.build_model()
        model4.load_weights('./hw3_models/model4.h5')
        models.append(model4)

        model5 = hw3_models.model5.build_model()
        model5.load_weights('./hw3_models/model5.h5')
        models.append(model5)

        model6 = hw3_models.model6.build_model()
        model6.load_weights('./hw3_models/model6.h5')
        models.append(model6)

        model7 = hw3_models.model7.build_model()
        model7.load_weights('./hw3_models/model7.h5')
        models.append(model7)

        model8 = hw3_models.model8.build_model()
        model8.load_weights('./hw3_models/model8.h5')
        models.append(model8)

        self.models = models
        self.generator = generator
        self.batch_size = batch_size
        self.mean = mean
        self.std = std

    def __next__(self):
        data = next(self.generator)
        label = self.predict(data)

        return data, label

    def predict(self, data):
        norm_1 = normalize_1(data)
        norm_2 = normalize_2(data, self.mean, self.std)

        num = 1
        label = self.models[0].predict(norm_1, batch_size = self.batch_size)

        for model in self.models[1:5]:
            label += model.predict(norm_1, batch_size = self.batch_size)
            num += 1

        for model in self.models[5:]:
            label += model.predict(norm_2, batch_size = self.batch_size)
            num += 1

        return np.clip(label/num, 0, 1)