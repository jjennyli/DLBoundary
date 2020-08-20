import os
from savemodel import SaveModelCallback
import numpy as np

from tensorflow.keras.models import load_model

class ModelComparator:
    def __init__(self):
        self.models = {}
        self.training_sets = {}
        self.testing_sets = {}
        self.histories = {}
        self.smc_params = { "filepath" : "checkpoints", "n" : 100 }
    
    def add_model(self, name, model):
        self.models[name] = model
    
    def add_training_set(self, name, data):
        self.training_sets[name] = data
        
    def add_testing_set(self, name, data):
        self.testing_sets[name] = data
        
    def set_smc_params(self, filepath, n):
        self.smc_params["filepath"] = filepath
        self.smc_params["n"] = n
        
    def train_models(self, **kwargs):
        for model_name in self.models:
            for tset_name in self.training_sets:
                print(f"MODEL {model_name} TRAINING ON DATASET {tset_name}")
                
                start_dir = self.smc_params["filepath"]
                fpath = f"{start_dir}/{tset_name}/{model_name}"
                nn = self.smc_params["n"]
                
                if not os.path.exists(start_dir):
                    os.makedirs(start_dir)
                if not os.path.exists(f"{start_dir}/{tset_name}"):
                    os.makedirs(f"{start_dir}/{tset_name}")
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                
                model = None
                checkpoints = [fpath + "/" + name
               for name in os.listdir(fpath)]
                if checkpoints:
                    latest_cp = max(checkpoints, key=os.path.getctime )
                    print('Restoring from', latest_cp)
                    model = load_model(latest_cp)
                
                if model == None:
                    model = self.models[model_name]()
                
                smc = SaveModelCallback(filepath=fpath, n=nn)
                if "callbacks" in kwargs:
                    kwargs["callbacks"].append(smc)
                else:
                    kwargs["callbacks"] = [smc]
                
                history = model.fit(
                    self.training_sets[tset_name][0], self.training_sets[tset_name][1],
                    **kwargs
                )
                
                self.histories[(model_name, tset_name)] = history
    
    def compare_min(self, metric):
        result = {}
        minimum = None
        min_name = None
        for hkey in self.histories:
            history = self.histories[hkey].history
            result[hkey] = np.amin(history[metric])
            if minimum == None or minimum >= result[hkey]:
                minimum = result[hkey]
                min_name = hkey
        return result, {"name" : min_name, "value":minimum}
            
    
    def compare_max(self, metric):
        result = {}
        maximum = None
        max_name = None
        for hkey in self.histories:
            history = self.histories[hkey].history
            result[hkey] = np.amax(history[metric])
            if maximum == None or maximum >= result[hkey]:
                maximum = result[hkey]
                max_name = hkey
        return result, {"name" : max_name, "value": maximum}
    
    def history_for(self, model_name, training_set_name):
        return self.histories[(model_name, training_set_name)]
    
    def test_models(self, use_training_sets=False, acc_range=0.5, use_decoder=None):
        if use_training_sets:
            testing_sets = self.training_sets
        else:
            testing_sets = self.testing_sets
        
        for model_name in self.models:
            for tset_name in testing_sets:
                print(f"\n\nMODEL {model_name} TESTING ON DATASET {tset_name}\n")
                model = self.models[model_name]()
                
                x_test = testing_sets[tset_name][0]
                y_test = testing_sets[tset_name][1]
                
                pred = model.predict(x_test)
                
                if use_decoder != None and model_name in use_decoder:
                    pred = np.argmax(pred)
                    pred = use_decoder[model_name][pred]
                
                test_total = 0
                acc_count = 0
                for a in range(len(x_test)):
                    x = x_test[a]
                    y = y_test[a]
                    yp = pred[a][0]
                    print(f"\tinput={x[0]} expected_output={[y]} prediction={yp} \n\t\tdifference={yp - y} \n\t\t%-different from actual={abs(yp - y)/y}")

                    if abs(yp - y) <= acc_range:
                        acc_count = acc_count + 1

                    test_total = test_total + 1

                    print(f"\t\t accuracy={acc_count / test_total}")