# Model
forward():
0. Gets padding mask for sentences
1. Get embeddings of the sentences
2. Get embeddings only for the [SEP] tokens (if self.use_sep is True)
2. Transform data such all sentences are as a one batch
3. If labels were provided (for training):
    * Check if they're equal to num_sentences
        * If not check with assert if they're bigger than num_sentences (it can happend because BERT crops the sentences) 
4. If labels were not provided just skip the part above
5. If there are some additional features along the text that you wanted to add, we concatenate it with the embedded sentences
6. Apply a linear layer to unwraped (time dimension) sequence of embeddings that are related to [SEP] token position. Its output is num of labels
7. If labels are not scores, get a softmax of these scores
8. Create output dictionary with losses and epoch metrics for the training (Pytorch specific)
9. Optionally add CRF layer (not needed for our project)
10. Reshape labels and predicted lables
11. Get the mean loss between those labels. Either MSE(if sci_sum==True) or Crossentropy.
12. Get the accuracy between the predicted label probabilities and the true labels (not probabilites). It uses CategoricalAccuracy as a metric
13. Gets F1 as a metric as well and returns output dict with a label loss and label logits

## Questions from the last week
### Are the weights initialized with TF Hub Model?
From https://www.tensorflow.org/hub/tf2_saved_model:
> Keras is TensorFlow's high-level API for building deep learning models by composing Keras Layer objects. The tensorflow_hub library provides the class hub.KerasLayer that gets initialized with the URL (or filesystem path) of a SavedModel and then provides the computation from the SavedModel, including its pre-trained weights.

So it seems that the BERT model taken from TF Hub should include the pretrained weights. By adding trainable=True, we are able to fine tune the weights.

> Creating a hub.KerasLayer like

> layer = hub.KerasLayer(..., trainable=True)

> enables fine-tuning of the SavedModel loaded by the layer. It adds the trainable weights and weight regularizers declared in the SavedModel to the Keras model, and runs the SavedModel's computation in training mode (think of dropout etc.).
The image classification colab contains an end-to-end example with optional fine-tuning.
### Which output should be taken from the BERT? 
From: https://github.com/google-research/bert/blob/master/run_classifier_with_tfhub.py
There are 2 different outputs: pooled output and token-level output. Pooled output returns embeddings for the CLS token. We, however, need the token-level output to get classification scores for each [SEP] token.
### What layers and how are put on top?
From the Pytorch code: 
We take the embeddings only that related to [SEP] tokens. On top of those embeddings, we put a TimeDistributed(Linear(ff_in_dim, self.num_labels)). This unwraps the LSTM time-wise and applies a Linear layer (MLP) for each embedding vector related to the location of [SEP] tokens. Then for our project, we need get a softmax layer on top. And that's it.
### How to use Segment ID mask?
I couldn't find code related to the segment mask in the Pytorch code. However, the top comment of the SegClassificationModel class says: 
> Question answering model where answers are sentences