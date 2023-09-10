# Pytorch Transformer Minimal Examples

A couple of minimal examples of how to use **[pytorch](https://pytorch.org/)** transformer. We consider a few (non-NLP) based problems to illustrate how transformer encoder and decoder layers can be used.

This code was produced for [my presentation](https://datasciencefestival.com/session/a-user-guide-to-transformers/) about using transformers at the [Data Science Festival Summer School 2023](https://datasciencefestival.com/event/summer-school/). I decided to give the talk after struggling myself to find simple examples. Most transformer tutorials seems to go directly from "transformers from scratch" to "use a pre-trained model" without explanation. The presentation and this code was made in an attempy to bridge that gap or get the community engaged to help bridge the gap for everyone wanting to learn about transformers.

If you find a way to make the code more readable/pedagogical, give better predictions or find some typo, please open a merge request. All contributions are welcome.



### Challenge :crown:

For each of these examples, what is the minimal (most pedagogical) network architecture with transformers (no pre-trained or rule-base components) that can be trained in reasonable time on a consumer grade laptop (CPU only) and achieve good performance (in a relevant metric for the problem)?

If there is no such transformer based model, what is the explanation?

Especially for the image_classification problem, we know that a rule-based model can achieve very high accuracy (build a standard network to classify the mnist images and map the label to the word) while my attempt here did not perform well.


## Sequence classification

We use the transformer-encoder to classify between 3 artificial "languages" as either a sequence (of symbols) or a timeseries (of floating point numbers). For more info, see this [README](./sequence_classification/README.md).

## Sequence generation

We use the transformer-decoder to complete sentences (as a sequence of characters) based on the works of Shakespeare. For more info, see this [README](./sequence_generation/README.md).

## timeseries_generation

Use the transformer-decoder to predict the future temperature from weather data. For more info, see this [README](./timeseries_generation/README.md).

## image_classification

Predict the written name (annotation) of a number based on an image. For more info, see this [README](./sequence_image_classification/README.md).