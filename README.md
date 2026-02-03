<img width="777" height="600" alt="image" src="https://github.com/user-attachments/assets/28a8324c-cc4d-4b80-bb39-02519cc2ef3a" /># Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.
<img width="1661" height="925" alt="image" src="https://github.com/user-attachments/assets/b6dc94a2-55f1-48cb-9ac3-36695fbf677d" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:JAGADEESH P

### Register Number:212223230083

```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss=criterion(ai_brain(X_train),y_train)
        loss.backward()
        optimizer.step()





        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```

### Dataset Information
<img width="919" height="623" alt="image" src="https://github.com/user-attachments/assets/09ab67da-0035-4840-a235-40a8c7cca89f" />


### OUTPUT
<img width="777" height="600" alt="image" src="https://github.com/user-attachments/assets/78e567b5-2880-41ef-80c2-7f5146b98264" />


### Training Loss Vs Iteration Plot
<img width="840" height="321" alt="image" src="https://github.com/user-attachments/assets/ac38a784-f9c4-456d-b300-fa9039c4de46" />


### New Sample Data Prediction
<img width="1078" height="191" alt="image" src="https://github.com/user-attachments/assets/ffce9f8d-424c-4cfb-86cc-daec582a8aff" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
