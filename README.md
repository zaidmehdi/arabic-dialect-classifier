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