# InformationRetrieval - Assignment 2

The system requires to install python3. In order to run the queries, you need 2 steps.

Firstly you need to run the prep.py code with the following command : python3 prep.py data_folder_path

data_folder_path is a path to Reuters-21578 data set.

After you run this command you can run your queries with the following command: python3 query.py your_query
your_query can be either a single word like ban or wildcard query like ban*. After that you will see the IDs of documents that contain your query in the ascending order.
Note that if you give a wildcard query you should give it in quotes like "ban*".