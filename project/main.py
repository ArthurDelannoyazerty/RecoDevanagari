from model_creation_parameters import ModelCreationParameters
from models_database import ModelsDatabase
from model_generator import ModelGenerator
from model_train_organisator import ModelTrainOrganisator


#load parameters
model_creation_parameters = ModelCreationParameters()
model_database = ModelsDatabase()
layers_models = model_database.layers_models

#load models
models = []
for i in range(len(layers_models)):
    layers_model = layers_models[i]
    model_generator = ModelGenerator()
    model_generator.create_model(layers_model)
    models.append(model_generator.model)
    model_generator.show_model(model_creation_parameters, i)

#train models
model_train_organisator = ModelTrainOrganisator(models)
model_train_organisator.train_all_models()
