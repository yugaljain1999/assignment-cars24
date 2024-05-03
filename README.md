# MNIST Image Classifier

* Training

    Refer notebooks/train.ipynb

* Inference

    Refer folder named 'classifier', here we have fastAPI deployed using docker and tested POST request on postman.

- Note - Weights are exported here - https://drive.google.com/file/d/1OAH7VyDif5xp1J_U_B_VNh1qMVnFwBPi/view?usp=sharing

- Download weights and place it under classifier/model

* Steps to build docker image and run container

    - Install docker

    - Run below commands

        - sudo docker build -t imageclassifier .
        - sudo docker run -d --name classifier -p 80:80 -i -t imageclassifier:latest

    - Open URL - http://0.0.0.0:80 in browser to run POST api for prediction.