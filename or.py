from os import PathLike
from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model,save_plot
import pandas as pd 
import numpy as np
OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1],
}

df = pd.DataFrame(OR)
print(df)

X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model_or = Perceptron(eta=ETA, epochs=EPOCHS)
model_or.fit(X, y)

_ = model_or.total_loss()

save_model(model_or,filename="or.model")
save_plot(df,"or.png",model_or)


