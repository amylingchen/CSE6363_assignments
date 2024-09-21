## directory structure
### 1.layer.py : Neural Network Library
- `Layer` Class 
-  `Linear` Layer Class
- `Sigmoid` Class
-  `ReLU` Class
- Binary Cross-Entropy Loss Class
- `Sequential` Class
- `L_layer_model_Classifier` Class 
- `L_layer_model_Regression` Class

### 2. utils.py: Utility Functions

### 3. solve the XOR problem.ipynb : 
Solve XOR problem with L_layer_model_Classifier**

    XOR_solved.w=array([[-3.98719196e-03, -1.47875726e-03],
                        [-3.49666014e+00,  3.55167242e+00],
                        [ 3.49242240e+00, -3.55287814e+00]])
### 4. Dataset Preprocessing.ipynb : 
Preprocess the dataset

### 5. Model Selection.ipynb: 
Predicting Trip Duration with three model

1. **model1**: one hidden layer and 7 notes with learning_rate = 1 and random_state = 12
   - trains 90 epochs 
   - score on the test set is 0.386 RMSLE 
   - ![model1_training_cost.png](images%2Fmodel1_training_cost.png)
   

  
2. **model2**:  one hidden layer and 4 notes with learning_rate = 0.7 and random_state = 12
   - trains 67 epochs 
   - score on the test set is 0.547 RMSLE 
   - ![model2_training_cost.png](images%2Fmodel2_training_cost.png)
   
3. **model3**:  two hidden layer and 7 notes and 2 notes with learning_rate=0.7 and random_state = 12
   - trains 81 epochs 
   - score on the test set is 0.443 RMSLE
   - ![model3_training_cost.png](images%2Fmodel3_training_cost.png)
