
# Flower Classification :



The following project can classify the Five different types of flowers with 90% Accuracy. The flowers which it can classify are Daisy, Dandelion, Rose, Sunflower and Tulip. This project is also deployed at hugging face, you can check it from the following link : [Flower Classification.](https://huggingface.co/spaces/khanaabidabdal/flower_classification)

The Dataset used for the training of Deep Learning Model is taken from the [Kaggle.](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) Here, I used a transfer learning technique for the development of the this application. The Model which I used is EfficientNetB0. Just I changed it's classifier to suit my problem and number of classes. Before finalizing EfficientNetB0 tried many different transfer learning models like VGG,EfficientNetB1, ResNet as well as my own build models. 


For development of this model, I wrote app.py, model.py, requirements.txt and downlaoded the state_dict of the trained model and saved it as the EfficientNet_Model. 

