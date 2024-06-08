
# Flower Classification :



The following project can classify the Five different types of flowers with 90% Accuracy. The flowers which it can classify are Daisy, Dandelion, Rose, Sunflower and Tulip. This project is also deployed at hugging face, you can check it from the following link : [Flower Classification.](https://huggingface.co/spaces/khanaabidabdal/flower_classification)


The dataset used for training the deep learning model was taken from Kaggle(https://www.kaggle.com/datasets/alxmamaev/flowers-recognition). I used a transfer learning technique for the development of this application. The model I used is EfficientNetB0. I changed its classifier to suit my problem and the number of classes. Before finalizing EfficientNetB0, I tried many different transfer learning models like VGG, EfficientNetB1, ResNet, as well as my own custom-built models. 


For development of this model, I wrote app.py, model.py, requirements.txt and downlaoded the state_dict of the trained model and saved it as the EfficientNet_Model. 

