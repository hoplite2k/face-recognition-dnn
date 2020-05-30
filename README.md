Face Recognition

This face recognition module is built using OpenCV and pre-trained caffe and torch models.

To build our face recognition system, we’ll first perform face detection, extract face embeddings from each face using deep learning, train a face recognition model on the embeddings, and then finally recognize faces in live video streams with OpenCV.

Even though the deep learning model has never seen the faces we’re about to pass through it, the model will still be able to compute embeddings for each face — ideally, these face embeddings will be sufficiently different such that we can train a “standard” machine learning classifier (SVM, SGD classifier, Random Forest, etc.) on top of the face embeddings, and therefore obtain our OpenCV face recognition pipeline.

Images of some famous personality is the dataset for this model. The pre_trained_models directory contains caffe and torch models. The embedding directory contains embeddings os the dataset which is obtained by executing "embedder.py". The label_encoder and recognizer directories contain the encoding for labels and recognizer model respectively which is obtained by executing "trainer.py". (The labels for images is obtained from the directory name of images present in the dataset).

In order to develop a face recognizer for your own dataset create a directory with the person's name and add some images of that person's images in that directory(min. of 6) and create a directory of unknow faces also. Then execute "embedder.py" and get the embeddings for the images in the dataset. These embeddings are saved in embeddings directory. Then we execute "trainer.py" so that to train the model on the dataset provided. This produces the encodings for labels and saves it in label_encoder and recognizer model which will be saved in recognizer directory. So our model is trained. Then we execute "recognizer.py" to start face recognition. This starts live video stream and starts detecting and recognizing the faces.

Execution:
1. Add your custom images in dataset directory and add some unknown images also(I have provided some images). 
2. Execute "python embedder.py" in terminal or command prompt.
3. Execute "python trainer.py" in terminal or command prompt.
4. Execute "python recognizer.py" in terminal or command prompt.

Done, the live stream is started and recognizing faces also.

Caution : If the recognizer doesn't produce proper output for some faces, then just increase the images in dataset.

REQUIREMENTS:
1. OpenCV 3.3 or above
2. imutils
3. sklearn
