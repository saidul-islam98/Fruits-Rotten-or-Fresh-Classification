import matplotlib.pyplot as plt
import os
from model import model_definition, model_summary
from optimizerandcallbacks import checkpoint_model, lr_reducer
from preprocess import train_validation_datagen
from sample_test import simple_test



def train_model(cnn_model, model_checkpoint_callback, reduce_lr, train_generator, validation_generator):
    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = cnn_model.fit(train_generator, epochs=25, validation_data = validation_generator, verbose = 1, validation_steps = 3, callbacks = [model_checkpoint_callback, reduce_lr])
    return cnn_model, history  


def plot_graphs(BASE, history, string, figname):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.savefig(os.path.join(BASE, figname))
  plt.show()



def main():
    BASE = './data'
    CHECKPOINT_BASE = './data/checkpoint'

    cnn_model = model_definition()
    model_summary(cnn_model)
    # model_plot(cnn_model)

    checkpoint_filepath, model_checkpoint_callback = checkpoint_model(CHECKPOINT_BASE)
    reduce_lr = lr_reducer()

    train_generator, validation_generator, class_labels = train_validation_datagen(BASE)
    cnn_model, history = train_model(cnn_model, model_checkpoint_callback, reduce_lr, train_generator, validation_generator)
    plot_graphs(BASE, history, 'accuracy', 'accuracy.png')
    plot_graphs(BASE, history, 'loss', 'loss.png')

    simple_test(BASE, class_labels, cnn_model)



main()