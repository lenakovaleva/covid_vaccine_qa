# covid_vaccine_qa
This is a question-answering system about COVID-19 vaccination.

We created a dataset using questions from official стопкоронавирус.рф site.

We use a simple unigram FastText model. It finds the most relevant question form the database using cosine similarity.

A simple Telegram Bot provides an interface for users. 
The Bot takes user question, find the most relevant question from the database and gives the found question and the answer.

Our system is created in Russian and devoted to helping Russian people find latest enhanced information about COVID-19.
