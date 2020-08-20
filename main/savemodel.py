from tensorflow.keras.callbacks import Callback

class SaveModelCallback(Callback):
    def __init__(self, filepath,  n=100, use_model_name=False):
        super(SaveModelCallback, self).__init__()
        self.num_epochs = n
        self.filepath = filepath
        if use_model_name:
            self.filepath = self.filepath + "/" + self.model.name
    
    def on_epoch_end(self, epoch, logs = None):
        super(SaveModelCallback, self).on_epoch_end(epoch, logs)
        if epoch % self.num_epochs == 0:
            self.model.save(f"{self.filepath}/{epoch}.model")
            print(f"model saved: {self.filepath}/{epoch}.model")