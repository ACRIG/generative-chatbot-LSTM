# Open-domain Generative-based Chatbot using Deep Learning and Sequence-to-sequence LSTM

- Model : Encoder-Decoder with Unidirectional LSTM
- Maximum Length : 15
- Hyperparameter:
    * Tuning method: Sequential
    * Embedding Dimension : 1536
    * Batch Size : 512
    * Hidden Dim LSTM : 512
    * Learning Rate : 0.001
 
## Deployment

To deploy this project run

### Step 1 : Build Docker Image
```bash
  docker build -t chatbot-app .
```

### Step 2 : Run Docker Container
```bash
  docker run -d --restart always --name flask-app -p 5000:5000 chatbot-app
```

To stop and delete the container run
```bash
  docker stop chatbot-app && docker rm chatbot-app
```
