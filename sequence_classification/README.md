# Sequence Classification with a Transformer Encoder Network

The network classifies a sequence of numbers by

1. Preprocessing the sequence

2. Adding positional encoding

3. Applying the transformer encoder

4. Pooling the output (mean pooling)

5. Applying a dense network to predict the label

## Token sequence classification

The standard NLP approach with transformers is to treat the sequence as a sequence of unordered characters. In this case the pre-processing is to embed the characters in a vectorspace.

## Timeseries sequence classification

For a sequence of ordered tokens (like integers), we can leverage the internal order by replacing the embedding with a convolutional layer. This approach results in a faster converging network.

TODO: Images