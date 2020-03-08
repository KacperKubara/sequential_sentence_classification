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