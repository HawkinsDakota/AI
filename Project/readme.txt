The scripts here present methods to perform object detection in scene images.

Group Members: Dakota Hawkins

To perform object detection, simply issue the following terminal command:

    > python object_localization.py /path/to/your/image

The scripts assume the following libraries are installed:
    
    1. PyTorch
    2. Numpy
    3. Sklearn
    4. Matplotlib
    5. Pandas
    6. Pickle
    7. PIL
    8. Skimage
    9. Seaborn

The helper scripts -- everything outside of object_localization.py -- exist
largely to build the AlexNet + SVM model used to detect objects in
object_localization.py. This model, along with a PCA model, are already
serialized in the pca_svm.pkl and pca.pkl file. 

Building the models yourself is very time and memory intensive, however, if you 
wish to do so, issue the following commands in order:

    > parse_data.py
    > alex_nn.py
    > perform_pca.py

This process assumes the class dataset is in the same working directory as the
scripts with 'dataset' as the name of the root directory.