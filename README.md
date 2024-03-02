# Arabic Dialect Classifier
This project is a classifier of arabic dialects at a country level:  
Given some arabic text, the goal is to predict the country of the text's dialect.  
  
You can use the "/classify" endpoint through a POST request with a json input of the form: '{"text": "Your arabic text"}'  
```
curl -X POST -H "Content-Type: application/json" -d '{"text": "Your Arabic text"}' http://localhost:8080/classify
```

## Run the app locally with Docker
1. Clone the repository with Git:  
```
git clone https://github.com/zaidmehdi/arabic-dialect-classifier.git
```
2. Build the Docker image:  
```
docker build -t adc .
```
3. Run the Docker Container:
```
docker run -p 8080:80 adc
```

Now you can try sending a POST request:
```
curl -X POST -H "Content-Type: application/json" -d '{"text": "Your Arabic text"}' http://localhost:8080/classify
```  
The response should be a json of the form:
```
{
    "class": "country_name"
}
```

## How I built this project:
The data used to train the classifier comes from the NADI 2021 dataset for Arabic Dialect Identification [(Abdul-Mageed et al., 2021)](#cite-mageed-2021).  
It is a corpus of tweets collected using Twitter's API and labeled thanks to the users location with the country and region.  

I used the language model `https://huggingface.co/moussaKam/AraBART` to extract features from the input text by taking the output of its last hidden layer. I used these word embeddings as the input for a Multinomial Logistic Regression to classify the input text into one of the 21 dialects (Countries).

For more detail, please refer to the docs directory.

## References
- <a name="cite-mageed-2021"></a>
[Abdul-Mageed et al., 2021](https://arxiv.org/abs/2103.08466)  
*Title:* NADI 2021: The Second Nuanced Arabic Dialect Identification Shared Task  
*Authors:* Abdul-Mageed, Muhammad; Zhang, Chiyu; Elmadany, AbdelRahim; Bouamor, Houda; Habash, Nizar  
*Year:* 2021  
*Conference/Book Title:* Proceedings of the Sixth Arabic Natural Language Processing Workshop (WANLP 2021)
