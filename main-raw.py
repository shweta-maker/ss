from flask import Flask, render_template, request, jsonify
import aiml
import os
import mnbpy
import logpy
import pickle
import pandas as pd
import csv
import smtplib, ssl, email
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from manual_spellchecker import spell_checker

mnb = pickle.load(open('mnb.pickle', 'rb'))
log = pickle.load(open('log.pickle', 'rb'))
count=0

def sendmail():

    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "test12mail98@gmail.com"  # Enter your address
    receiver_email = "rudvi306@gmail.com"  # Enter receiver address
    password = "testmail!@98"

    messagemail = MIMEMultipart()
    messagemail['Subject'] = "Admission enquiry"

    with open('newdata.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            body_part = MIMEText(row[0]+"\n", 'plain')
            messagemail.attach(body_part)
        read_obj.close()

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, messagemail.as_string())

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('chat.html')

@app.route("/ask", methods=['POST'])
def ask():

    message =request.form['messageText'].strip()

    kernel = aiml.Kernel()

    if os.path.isfile("bot_brain.brn"):
        kernel.bootstrap(brainFile = "bot_brain.brn")
    else:
        kernel.bootstrap(learnFiles = os.path.abspath("aiml/startup.xml"), commands = "load aiml b")
        kernel.saveBrain("bot_brain.brn")

    while True:
        if message == "quit":
            exit()
        elif message == "save":
            kernel.saveBrain("bot_brain.brn")
        else:

            bot_response = kernel.respond(message)
            if (bot_response == "NULL"):

                testa=mnbpy.clean_text(message)
                if(len(testa.split())>1):
                    X_test=pd.Series(testa)
                    y_prednb = mnb.predict(X_test)
                    y_predlog = log.predict(X_test)

                    if(y_prednb==y_predlog and mnb.predict_proba(X_test).max()>0.5):
                        csv_f = csv.reader(open('answers.csv',encoding='utf-8'))

                        for row in csv_f:
                            if(row[0]==str(y_prednb[0])):
                                gt=row[1]
                                print (row[1])

                    else:
                        global count
                        count= count+1
                        gt="Please provide more information"
                        with open('newdata.csv', 'a', newline="") as file:
                            csv_w = csv.writer(file)
                            csv_w.writerow([message])
                            f.close()
                        df = pd.read_csv("newdata.csv")
                        ob = spell_checker(df, "text")
                        ob.correct_words()
                     
                        
                        with open('newdata.csv') as check:
                            csv_r = csv.reader(check)
                            if(len(list(csv_r))>=10):
                                sendmail()
                                check.close()
                                os.remove("newdata.csv")

                        if(count>2):
                            gt="Please mail your query to sies@gmail.com"
                            count=0
                else:
                    gt="Please provide more information"



                return jsonify({'status':'OK','answer':'$'+gt})


            else:
                return jsonify({'status':'OK','answer':bot_response})


if __name__ == "__main__":
    app.run(host='localhost', debug=True)
