# Nested Named Entity Recognition (NNER)

This is a project for the DD2417 course.

We used the GENIA project term annotation corpus to train different ML models to recognize nested named entities.


# Code Components

This project consists of the componenets enumerated below. Also see instructions to run each component.

# 1. Dataset Creation
  We use the GENIA corpurs to create the training and testing data for our vectors and classification model. Run the file 'data_manipulation.py' to geenrate the testing and training datasets in the data folder. These files have already been created and are present in the folder 'data'.
  
# 2. Creation of Random Indexing Vectors Binary Logistic Regression Classifier Models
   RandomIndexing (RI) Vectors need to be created if not already present. Change the variable 'vector_flag' equal to 'ri'. Run the files 'ClassificationModel1gram.py', 'ClassificationModel2gram.py' and 'ClassificationModel2gram.py' to generate vectors and models on the testing dataset created in the previous step. The models and ri_vectors have already been created and are present in the below files:
   - RandomIndexing Vectors 1-gram model : ri_vectors.txt
   - RandomIndexing Vectors 2-gram model : ri_vectors2.txt
   - RandomIndexing Vectors 3-gram model : ri_vectors3.txt
   - Classification Model 1-gram using RI vectors : model_2022-05-18_19_19_23_477034/b.model
   - Classification Model 2-gram using RI vectors : model_2022-05-18_19_23_19_866869/b2.model
   - Classification Model 3-gram using RI vectors : model_2022-05-18_19_36_12_387203/b3.model
   
# 3. Creation of Random Indexing Vectors Binary Logistic Regression Classifier Models
   In order to use the manually created features for classification of named entities we much create these vectors and their corresponding classification models. Change the variable 'vector_flag' equal to 'manual'. Run the file 'CreateVectors.py' to generate the manual word vectors for training and testing data. Run the files 'ClassificationModel1gram.py', 'ClassificationModel2gram.py' and 'ClassificationModel2gram.py' to generate models. The models and manual_vectors have already been created and are present in the below files:
   - Manual Vectors 1-gram model : data/X_feature_vectors_v2/training_X1_v2 and data/X_feature_vectors_v2/testing_X1_v2
   - Manual Vectors 2-gram model : data/X_feature_vectors_v2/training_X2_v2 and data/X_feature_vectors_v2/testing_X2_v2
   - Manual Vectors 3-gram model : data/X_feature_vectors_v2/training_X3_v2 and data/X_feature_vectors_v2/testing_X3_v2
   - Classification Model 1-gram using manual vectors : model_2022-05-19_12_34_23_589091/b_manual.model
   - Classification Model 2-gram using manual vectors : model_2022-05-19_12_35_40_008350/b2_manual.model
   - Classification Model 3-gram using manual vectors : model_2022-05-19_12_38_11_106153/b3_manual.model
   
# 4. Running the NNER Tool
   The user interface to run the NNER tool is launched by running the file 'NNER.py'. Please note that if all the above the above mentioned files are alreeady created and in-place, for-example when downloading this repository, one needs to only run this file in order to interact with the tool.
    
    
 # Note: 
 If any new file is created using steps 1 to 3, the user needs to make updates to file names in the main methods of 'ClassificationModel1gram.py', 'ClassificationModel2gram.py', 'ClassificationModel2gram.py' and 'NNER.py'.
