from keras.models import clone_model
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from time import time
import TensorBoardHelper

"""
:param model: The keras model to find optimal hyperparameters for.
:param hyperparameters: An instance of HyperparameterGenerator
:param train_x: The training data to train on
:param train_y: The true labels of the training data.
"""
class ModelOptimizer:
    def __init__(self, model, hyperparameter_generator, train_x, train_y):
        self.original_model = model
        self.hyperparameter_generator = hyperparameter_generator
        self.train_x = train_x
        self.train_y = train_y
        self.results = []


    def Optimize(self):
        #Launch tensorboard
        TensorBoardHelper.run()
        #Iterate over the different hyperparameters
        hp = next(self.hyperparameter_generator)
        while(hp != None):
            print("Val split: " + str(hp.validation_split) + " batch: " + str(hp.batch_size))
            # Clone the original model to make sure it is not changed over the attempts.
            model = clone_model(self.original_model)
            # Compile and fit with the given parameters
            model.compile(hp.optimizer, loss=hp.loss, metrics=hp.metrics)
            history = model.fit(self.train_x, self.train_y, batch_size=hp.batch_size, epochs=hp.epochs, verbose=2, validation_split=hp.validation_split, callbacks=hp.callbacks)
            #Add the results to the report list
            self.results.append(Report(model, hp, history.history['val_acc'][-1]))
            hp = next(self.hyperparameter_generator)
        #Sort the list in accordance to the validation accuracy and return all atempts
        self.results.sort(key=lambda report: report.accuracy, reverse=True)
        return self.results

'''
The report class is a simple data structure which stores a keras model as well as how it was trained and how it performed.
'''
class Report:
    def __init__(self, model, hyperparameters, accuracy):
        self.model = model
        self.hyperparameters = hyperparameters
        self.accuracy = accuracy

    def __str__(self):
        return "Accuracy: " + str(self.accuracy) +" Hyperparameters: {" + str(self.hyperparameters) + "} Model: {" + str(self.model) + "}"

"""
A hyperparameter generator which will create HyperparameterSets used by the optimizer to iterate through the different hyperparameters.
"""
def HyperparameterGenerator(loss, validation_split = [0.1, 0.2], batch_size = [16, 64, 128, 256, 1024], epochs = 100000, optimizer = 'adam', metrics = ['accuracy'], early_stopping = True):
        if (loss == None):
            raise ValueError('Model can not be compiled without a loss function. See https://keras.io/losses/ for a selection of loss functions.')
        for validation_split in validation_split:
            temp = batch_size.copy()
            for batch in temp:
                yield HyperparameterSet(loss, validation_split, batch, epochs, optimizer, metrics, early_stopping)
        yield None

"""

"""
class HyperparameterSet:
    def __init__(self, loss, validation_split, batch_size, epochs, optimizer, metrics, early_stopping):
        self.loss = loss
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.metrics = metrics
        self.early_stopping = early_stopping
        self.callbacks = [TensorBoard(log_dir="logs/{}".format(time())), ReduceLROnPlateau()]
        if early_stopping:
            self.callbacks.append(EarlyStopping(patience=5))
    
    def __str__(self):
        return "loss: " + str(self.loss) + ", validation_split: " + str(self.validation_split) +  ", batch_size: " + str(self.batch_size) + ", epochs: " + str(self.epochs) + ", optimizer: " + str(self.optimizer) +", metrics: " + str(self.metrics) + ", early_stopping: " + str(self.early_stopping) + ", callbacks: " + str(self.callbacks)