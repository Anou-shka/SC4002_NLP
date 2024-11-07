## 3.3. BiLSTM and BiGRU

<center>
    <img src = "./image/LSTM_GRU.jpg" width = 40%>  &emsp;  <img src = "./image/MLP.jpg" width = 20%>
    <br>
    Figure 1. Image for biLSTM and biGRU for sentiment analisys
</center>

<br>

&emsp; In this section, we improved our model by introducing BiLSTM and BiGRU to our structure. Our structure is shown in Fig. 1. First of all, we use the Tokenizer and Embedding model defined in Part 1 to tokenize the input sentence into a word sequence and generate corresponding embedding vectors. Secondly, We use bi-directional GRU and LSTM to encode the input sequence into hidden vectors. To associate the hidden vectors from two directions, we concate the hidden vectors from each direction. Thirdly, to consider hidden vectors from all positions, we summate the hidden vectors from all positions to a summated hidden vector. Finally, an MLP module with a sigmoid activation function maps the summated hidden states to a scalar output **y** ranging from 0 to 1, which is considered as the possibility of the sentence to be positive.

&emsp; We train a BiLSTM and BiGRU separately. We use binary cross-entropy loss as our criteria, and the model is used as a binary classifier. We use AdamW as an optimizer, which is a common choice in NLP tasks. To dynamically tune the learning rate to escape the local minimum, we use a cosine scheduler with warm-up as a learning rate scheduler. 

&emsp; We define the biGRU model with 2 GRU layers for each direction; the hidden dimension of the biGRU model is 256. As shown in the right part of Fig. 1, The MLP structure is a three-layer neural network with a GELU activation function and a two-layer normalization module. The output activation function is a scaler of binary classification. The biLSTM model is similar to the biGRU model, with only the GRU layer replaced by the LSTM layer. 

&emsp; We adopted the mini-batch mechanism in our training process. However, the sentence length varies in each batch. To handle variable sentence lengths within each batch, we dynamically pad each sentence to match the length of the longest sentence in the mini-batch. This approach saves GPU memory compared to global padding, as sentences do not need to be padded to the maximum length of the training set. Additionally, we mask padding positions to exclude them from backpropagation, ensuring they do not affect gradient updates.

&emsp; The setting of training process is below (we use same setting for both biGRU and biLSTM):

- dropout rate: 0.1
- training epochs: 3
- batch size: 32
- learning rate (before schedule): 8e-5
- weight decay rate: 1e-5
- warm-up ratio: 0.1
- maximum gradient norm: 2.0

Dropout is applied to both the biLSTM/biGRU and MLP parts to enhance generalizability. We use mini-batches of size 32, with an initial learning rate of 8e-5 that the scheduler adjusts dynamically during training to help the model avoid local minima. This learning rate is the maximum rate during the warm-up phase of the scheduler, with warm-up lasting for the first 10% of the total training steps. During warm-up, the learning rate gradually increases to 8e-5 and gradually declines following a cosine decay schedule after that, enabling the model to capture basic patterns in the early stages. To prevent gradient explosion, we apply gradient clipping with a maximum gradient norm of 2.0 after each backpropagation step.

<center>
    <img src = "./image/Training_curve_LSTM.png" width = 60%>
    <br>
    Figure 2. Training curve for biLSTM
</center>

<center>
    <img src = "./image/Training_curve_GRU.png" width = 60%>
    <br>
    Figure 3. Training curve for biGRU
</center>


<!-- 

Note: we may add more explaination if there are still some place left

-->

&emsp; As the training epochs is 3, in order to further track the training process, we compute loss and accuracy on the validation set every 100 steps. The outcome training curve is shown in Fig. 2. (biLSTM) and Fig. 3. (biGRU). The accuracy for biLSTM and biGRU on the test set is shown below:

- biGRU: 0.7486

- biLSTM: 0.7364

Compared with the RNN model, the biGRU and the biLSTM perform better. This may be because the GRU and LSTM modules introduce gate mechanisms to RNN-base models. The mechanism eases the gradient vanish issue of the RNN-base model. Moreover, biGRU and biLSTM are bi-directional. This means that each position can encode information for the whole sequence from both sides compared with the single-directional RNN model we defined previously. Comparing biGRU and biLSTM, the accuracy of the test set is similar. biGRU may be slightly better, but generally speaking, biGRU and biLSTM obtain similar performance.