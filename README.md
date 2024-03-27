---
title: Arabic Dialect Classifier
emoji: 🐪
colorFrom: yellow
colorTo: yellow
sdk: docker
app_port: 8080
license: mit
pinned: false
---

# Arabic Dialect Classifier
This project is a classifier of arabic dialects at a country level:  
Given some arabic text, the goal is to predict the country of the text's dialect.
  
[Link to the Demo](https://huggingface.co/spaces/zaidmehdi/arabic-dialect-classifier)
  
![Demo App](docs/images/gradio_app.png "Demo App")
## Run the app locally with Docker:
1. Clone the repository with Git:  
```
git clone https://github.com/zaidmehdi/arabic-dialect-classifier.git
```
2. Build the Docker image:  
```
sudo docker build -t adc .
```
3. Run the Docker Container:
```
sudo docker run -p 8080:8080 adc
```
  
Now you can access the demo locally at:
```
http://localhost:8080
```

## How I built this project:
The data used to train the classifier comes from the NADI 2021 dataset for Arabic Dialect Identification [(Abdul-Mageed et al., 2021)](#cite-mageed-2021).  
It is a corpus of tweets collected using Twitter's API and labeled thanks to the users' locations with the country and region.  

In the current version, I finetuned the language model `https://huggingface.co/moussaKam/AraBART` by attaching to it a classification head and freezing the weights of the base model (due to compute constraints):
```
(classification_head): MBartClassificationHead(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (dropout): Dropout(p=0.0, inplace=False)
    (out_proj): Linear(in_features=768, out_features=21, bias=True)
)
```
The model classifies any input text into one of the 21 countries that we have in the dialects dataset.
Currently, it achieves an accuracy of 0.357 on the test set.

For more details, you can refer to the docs directory.

## Releases
### v0.0.2
In the second release, I finetuned the langage model `https://huggingface.co/moussaKam/AraBART` by attaching to it a classification head and freezing the weights of the base model (due to compute constraints):
```
(classification_head): MBartClassificationHead(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (dropout): Dropout(p=0.0, inplace=False)
    (out_proj): Linear(in_features=768, out_features=21, bias=True)
)
```
**Accuracy achieved on test set: 0.357**


### v0.0.1
In the first release, I used the language model `https://huggingface.co/moussaKam/AraBART` to extract features from the input text by taking the output of its last hidden layer. I used these vector embeddings as the input for a Multinomial Logistic Regression to classify the input text into one of the 21 dialects (Countries).
  
**Accuracy achieved on test set: 0.2324**

## References:
- <a name="cite-mageed-2021"></a>
[Abdul-Mageed et al., 2021](https://arxiv.org/abs/2103.08466)  
*Title:* NADI 2021: The Second Nuanced Arabic Dialect Identification Shared Task  
*Authors:* Abdul-Mageed, Muhammad; Zhang, Chiyu; Elmadany, AbdelRahim; Bouamor, Houda; Habash, Nizar  
*Year:* 2021  
*Conference/Book Title:* Proceedings of the Sixth Arabic Natural Language Processing Workshop (WANLP 2021)
