# CS543 Massive Data Storage and Retrieval and Deep Learning (Project 2)

## Total_Outbound_Flights_Prediction

#### Team members: Arwa El-Hawwat, Jaini Patel, Rahul Dev Ellezhuthil. (Group-6)

### About:

In this project, we have used the Airline Data available from 2009 Data Expo - Airline On-Time Performance to develop a tool which predicts the total number of possible out-bound flights on a given day in the year at a particular airport, using Deep Learning. The dataset has 40M records and is approximately 1.5 GB in size and has 4 supporting csv files which have information of airport, carriers, planes and metadata of the dataset. The tools and technologies used in this project are Pytorch, Flask, HTMl and CSS.

### Goal:

Our main objective for this project is to develop a model based on pre-existing flight data in the United States to output the probable number of outbound flights scheduled from a particular airport on a specific date based on a variety of input factors (including date of travel and origin) and to be able to answer the following questions: What is the total possible number of outbound flights from a specific airport on a given date? Thus, which airport is busier at a given period of time in the year? After preprocessing the data, we feed the resulting data into our model to be encoded using embeddings which are then trained to achieve the ideal model based on our loss function. The resulting output is the number of outbound flights predicted to be scheduled on a particular sequence of input parameters. We aim to be able to organize and display our findings in a simple process and web application model.

Expected Users of this tool are, Airline staff, route planners, pilots and US domestic travellers.

Steps to run:

  1.  git clone https://github.com/Jaini8/CS543_Total_Outbound_Flights_Prediction.git
  2.  python app.py

Run : http://ilab1.cs.rutgers.edu:9995/

Video of the working: [Total Outbound Flights Prediction](https://www.youtube.com/watch?v=INmbzbuPgpg)
