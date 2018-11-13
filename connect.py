import pymongo
import configparser

config = configparser.ConfigParser()
config.read("config.ini")
user = config.get("Credentials", "User")
password = config.get("Credentials", "Password")

connectString = "mongodb+srv://{}:{}@cluster0-yhwvj.gcp.mongodb.net/test?retryWrites=true".format(user,password)
client = pymongo.MongoClient(connectString)
print(client.list_database_names())
