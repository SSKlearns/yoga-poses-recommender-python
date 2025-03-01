# Yoga pose recommender
A Flask Web Application that demonstrates how to do Vector Search on a Yoga Poses database in MongoDB Atlas. 

The application is a step by step demonstration of the following:
1. Utilize an existing Hugging Face Dataset of Yoga poses (JSON format).
2. Enhance the dataset with an additional field `description` that uses Gemini to generate descriptions for each of the poses.
3. Use Langchain to create a Document, use MongoDB Langchain integration to create the collection in Mongo.
4. Create a Atlas Vector Search Index to allow for Vector search.
5. Utilize the Vector Search in a Flask Application that brings everything together as shown below:

<img title="a title" alt="Alt text" src="/images/screenshot.png">

## Technologies used:
1. Python + Flask Framework
2. Google Cloud : Cloud Run, Gemini in Vertex AI
3. Langchain Python framework
4. MongoDB Atlas

## Codelab
Step by Step instructions to understand and run/deploy this application is available in this [codelab](https://codelabs.developers.google.com/yoga-pose-firestore-vectorsearch-python?hl=en#0).
