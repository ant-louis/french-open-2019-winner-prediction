# Setup
    conda create -n bigdataproject python=3.7
    source activate bidataproject
    conda install pymongo

# How to use connect and use the database
1. Create a new mongoDB Atlas user
2. Create a new cluster user with the link I emailed you
3. Set your IP as whitelist 
4. [How to connect using python](https://docs.atlas.mongodb.com/driver-connection/#python-driver-example)
and you can also see how to do this in the `connect.py` script I created

To use your own password and user, create a in your local folder a config file like `exampleconfig.ini` and if it has a different name than `config.ini`, add it to the .gitignore textfile to prevent it from being uploaded publicly to this repository

More info here : [MongoDB Atlas Documention](https://docs.atlas.mongodb.com/connect-to-cluster/)
