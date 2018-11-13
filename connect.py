import pymongo
import configparser
import pprint

config = configparser.ConfigParser()
config.read("config.ini")
user = config.get("Credentials", "User")
password = config.get("Credentials", "Password")

connectString = "mongodb+srv://{}:{}@cluster0-yhwvj.gcp.mongodb.net/test?retryWrites=true".format(user,password)
client = pymongo.MongoClient(connectString)
db = client.Tournaments
collection = db.Tournament2011
pprint.pprint(collection.find())