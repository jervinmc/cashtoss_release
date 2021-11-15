from flask import Response
from flask import Flask, jsonify, request,redirect
from flask_restful import Resource, Api
from flask_cors import CORS
#Imports
import string
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktSentenceTokenizer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk.classify
from sklearn.svm import LinearSVC
import requests
from datetime import datetime
from functools import wraps
import time
from Database import Database
import boto3
import os
from decouple import config
now = datetime.now().date()
app=Flask(__name__)
CORS(app)
api=Api(app)
#NLTK Downloads (Need to do only once)
nltk.download('punkt') 
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet') 
nltk.download('nps_chat')

#Global Constants
GREETING_INPUTS    = ("hello", "hi")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "Talkin' to me?"]
FILENAME           = "medical_faq.txt"
#Global Variables
lem = nltk.stem.WordNetLemmatizer()
remove_punctuation = dict((ord(punct), None) for punct in string.punctuation)

#Functions
'''
fetch_features transforms a chat into a classifier friendly format
'''
def fetch_features(chat):
    features = {}
    for word in nltk.word_tokenize(chat):
        features['contains({})'.format(word.lower())] = True
    return features
'''
lemmatise performs lemmatization on words
'''
def lemmatise(tokens):
    return [lem.lemmatize(token) for token in tokens]
  
'''
tokenise tokenizes the words
'''
def tokenise(text):
    return lemmatise(nltk.word_tokenize(text.lower().translate(remove_punctuation)))

'''
Standard greeting responses that the bot can recognize and respond with
'''
def greet(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
'''
match matches a user input to the existing set of questions
'''


def match(user_response):
    resp      =''
    q_list.append(user_response)
    TfidfVec  = TfidfVectorizer(tokenizer=tokenise, stop_words='english')
    tfidf     = TfidfVec.fit_transform(q_list)
    vals      = cosine_similarity(tfidf[-1], tfidf)
    idx       = vals.argsort()[0][-2]
    flat      = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if(req_tfidf==0):
        resp = resp+"not recognize"
        return resp
    else:
        resp_ids = qa_dict[idx]
        resp_str = ''
        s_id = resp_ids[0]
        end = resp_ids[1]
        while s_id<end :
            resp_str = resp_str + " " + sent_tokens[s_id]
            s_id+=1
        resp = resp+resp_str
        return resp


#Training the classifier
chats = nltk.corpus.nps_chat.xml_posts()[:10000]
featuresets = [(fetch_features(chat.text), chat.get('class')) for chat in chats]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.classify.SklearnClassifier(LinearSVC())
classifier.train(train_set)
# classifier = nltk.NaiveBayesClassifier.train(train_set) #If you need to test Naive Bayes as well
# print(nltk.classify.accuracy(classifier, test_set))

# #Question Bank Creation
ques_bank   = open(FILENAME,'r',errors = 'ignore')
qb_text     = ques_bank.read()
qb_text     = qb_text.lower()
sent_tokens = nltk.sent_tokenize(qb_text)# converts to list of sentences 
word_tokens = nltk.word_tokenize(qb_text)# converts to list of words
print(sent_tokens)
qa_dict     = {} #The Dictionary to store questions and corresponding answers
q_list      = [] #List of all questions
s_count     = 0  #Sentence counter

#Extract questions and answers
#Answer is all the content between 2 questions [assumption]
while s_count < len(sent_tokens):
    result = classifier.classify(fetch_features(sent_tokens[s_count]))
    # print(fetch_features(sent_tokens[s_count]))
    if("question" in result.lower()):
        next_question_id = s_count+1
        next_question = classifier.classify(fetch_features(sent_tokens[next_question_id]))
        while(not("question" in next_question.lower()) and next_question_id < len(sent_tokens)-1):
            next_question_id+=1
            next_question = classifier.classify(fetch_features(sent_tokens[next_question_id]))
        q_list.append(sent_tokens[s_count])
        
        end = next_question_id
        if(next_question_id-s_count > 5):
            end = s_count+5
        qa_dict.update({len(q_list)-1:[s_count+1,end]})
        s_count = next_question_id
    else:
        s_count+=1
        
#Response Fetching
class Chatbot(Resource):
    def post(self): #method.
        res = request.get_json()
        value = res.get('value')
        value = value.split("\n")
        print(value)
        for x in value:
            print(x)
            u_input = x
            u_input = u_input.lower()
            if(u_input!='ciao'):
                predict = match(u_input).strip().capitalize()
                q_list.remove(u_input)
                if(predict.replace('.','')=='Medication'):
                    return {"data":predict.replace('.','')}
                if(predict.replace('.','')=='Education'):
                    return {"data":predict.replace('.','')}
                if(predict.replace('.','')=='Food'):
                    print("food")
                    return {"data":predict.replace('.','')}
                if(predict.replace('.','')=='Utilities'):
                    return {"data":predict.replace('.','')}

        return {"data":"Others"}


class Usermanagement(Resource):
    def __init__(self):
        self.db=Database()

    def post(self,pk=None):
        data = request.get_json()
        try:
            self.db.insert(f"INSERT INTO users(email,password) values('{data.get('email')}','{data.get('password')}')")
            return {"status":"success"}
        except Exception as e:
            print(e)
            return {"status":"Failed Input"}

    def get(self,pk=None):
        if pk==None:
            res = self.db.query('SELECT * FROM users')
        else:
            res = self.db.query(f'SELECT * FROM users where id={pk}')
        return {"data":res}

    def delete(self,pk):
        try:
            self.db.insert(f'DELETE FROM users where id={pk}')
            return {"data":"success"}
        except:
            return {"status":"Failed"}
    
    def put(self,pk):
        data = request.get_json()
        try:
            self.db.insert(f"UPDATE users set email='{data.get('email')}',password='{data.get('password')}' where id={pk}")
            return {"status":"Success"}
        except Exception as e:
            return {"status":"Failed"}

class Categories(Resource):
    def __init__(self):
        self.db=Database()
        self.listitem=[]
    def get(self,category=None):
        print(category)
        try:
            res = self.db.query(f"select * from receipt where categories='{category}' ")
            if(res==[]):
                print("okay")
                return Response({"status":"Wrong Credentials"},status=404)
            else:
                print(res)
                listitem=[{"vendor_name":i[2],"created_date":i[3],"category":i[4],"total":i[5],"image":f"https://cashtoss.s3.ap-southeast-1.amazonaws.com/{i[6]}"} for i in res]
                print(listitem)
                return listitem
        except Exception as e:
            print(e)
            return {"status":f"{e}"}


class Receipt(Resource):
    def __init__(self):
        self.db=Database()
    def get(self):
        item={"total":0.0,"Medication":0.0,"Groceries":0.0,"Others":0.0,"Food":0.0,"Transportation":0.0,"Education":0.0}
        query = self.db.query(f"SELECT categories,sum(total) FROM receipt group by categories")
        total = self.db.query(f"SELECT SUM(total) FROM receipt")
        # item =  [{f"{x[0]}":float(x[1]) for x in query}]
        for x in query:
            if(x[0]=='Medication'):
                item['Medication']=float(x[1])
            if(x[0]=='Groceries'):
                item['Groceries']=float(x[1])
            elif(x[0]=='Others'):
                item['Others']=float(x[1])
            elif(x[0]=='Food'):
                item['Food']=float(x[1])
            elif(x[0]=='Education'):
                item['Education']=float(x[1])
            elif(x[0]=='Utilities'):
                item['Utilities']=float(x[1])
        item['total']=total[0][0]
        return item

    def post(self,pk=None):
        data = request.get_json()
        try:
            res = self.db.insert(f"INSERT INTO receipt values(default,'{data.get('id')}','{data.get('vendor_name')}','{now}','{data.get('category_name')}',{data.get('total')})")
            if(res==[]):
                print(res)
                return Response({"status":"Wrong Credentials"},status=404)
            else:
                result_data = self.db.query(f"SELECT SUM(total) FROM receipt")
                result_settings = self.db.query(f"SELECT totalAmount from settings")
                id = self.db.query(f"select max(id) from receipt")
                print(id[0])
                if(int(result_data[0][0])>=(result_settings[0][0])):     
                    print(result_settings[0][0])    
                    return {"status":"exceed","id":id[0][0]}
                else:
                    return {"status":"less","id":id[0][0]}
                
        except Exception as e:
            print(e)
            return {"status":f"{e}"}





class Login(Resource):
    def __init__(self):
        self.db=Database()

    def post(self,pk=None):
        data = request.get_json()
        print(data)
        try:
            res = self.db.query(f"SELECT * FROM users where email='{data.get('email')}' and password='{data.get('password')}'")
            if(res==[]):
                print(res)
                return Response({"status":"Wrong Credentials"},status=404)
            else:
                return Response({"status":"success"},status=201)
            
        except Exception as e:
            print(e)
            return {"status":"Failed Input"}

class Settings(Resource):
    def __init__(self):
        self.db=Database()

    def get(self,pk=None):
        result_settings = self.db.query(f"SELECT totalAmount from settings")
        return result_settings[0][0]

    def post(self):
        res = request.get_json()
        print(res)
        try:
            self.db.query(f"UPDATE settings set totalAmount={res.get('totalAmount')}")
            return {}
        except:
            return {}

class Upload(Resource):
    def __init__(self):
        self.db=Database()

    def post(self,pk=None):
        imageFile=request.files['image']
        file_path=os.path.join('', imageFile.filename) # path where file can be saved
        imageFile.save(file_path)
        client = boto3.client('s3',aws_access_key_id=config("AWS_ACCESS_ID"),aws_secret_access_key=config("AWS_SECRET_ID"))
        client.upload_file(f'{imageFile.filename}','cashtoss',f'{imageFile.filename}')
        print(pk)
        self.db.insert(f"UPDATE receipt set image='{imageFile.filename}' where id={pk} ")
        # data = request.get_json()
        # print("open pin 1")
        return {"status":"Successful"}


# class UploadTest(Resource):
#     def __init__(self):
#         self.db=Database()

#     def post(self,pk=None):
#         file = request.files['file']
#         file_path=os.path.join('', file.filename) # path where file can be saved
#         file.save(file_path)
#         client = boto3.client('s3',aws_access_key_id="AKIA5HVDPP5SGQIJIXO3",aws_secret_access_key="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")
#         client.upload_file(f'{file.filename}','comappt',f'{file.filename}')
#         return {"filename":f"https://s3.console.aws.amazon.com/s3/upload/comapptpublic?region=us-east-2/{file.filename}"}

api.add_resource(Usermanagement,'/api/v1/users/<int:pk>')
api.add_resource(Login,'/api/v1/login')
api.add_resource(Chatbot,'/api/v1/chat')
api.add_resource(Receipt,'/api/v1/receipt')
api.add_resource(Settings,'/api/v1/settings')
api.add_resource(Upload,'/api/v1/upload/<int:pk>')
# api.add_resource(UploadTest,'/api/v1/uploadtest')
api.add_resource(Categories,'/api/v1/categories/<string:category>')
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')