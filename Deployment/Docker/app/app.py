from flask import Flask,send_from_directory, flash, request, redirect, render_template, jsonify
import os      # For File Manipulations like get paths, rename
from werkzeug.utils import secure_filename
import urllib.request
import json
import os.path
import platform
from datetime import datetime
from io import BytesIO
import pandas as pd
from Reason_for_the_call import reason_for_the_call
from Action_item import action_item
from Sentiment_and_outcome import processing, sentiment_of_the_call, status_of_the_call
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
reason_model_args = ClassificationArgs(sliding_window=True)
reason_model_args.num_train_epochs = 10

reason_intent_model_path='../Downloads/reason_intentclassification/checkpoint-250-epoch-10'

reason_classification_model = ClassificationModel(
    "bert", reason_intent_model_path,args=reason_model_args,use_cuda=False
)
reason_bart_model_path="../Downloads/Reason_Bart/checkpoint-130-epoch-10"

reason_bart_model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name=reason_bart_model_path,use_cuda=False
    )


action_model_args = ClassificationArgs(sliding_window=True)
action_model_args.num_train_epochs = 10

action_intent_model_path='../Downloads/action_intentclassification/checkpoint-138-epoch-6'

action_classification_model = ClassificationModel(
    "bert", action_intent_model_path,args=action_model_args,use_cuda=False
)
action_bart_model_path="../Downloads/Bart_ActionItems/checkpoint-165-epoch-15"

action_bart_model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name=action_bart_model_path, use_cuda=False
    )

import openai
openai.api_key = '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'

status_model_name="curie:ft-personal-2022-03-06-10-17-39"
sentiment_model_name="curie:ft-personal-2022-03-06-09-52-08"

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/Call_summary', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        resp = jsonify({'message' : 'No file part in the request'})
        resp.status_code = 400
        return resp
    file = request.files['file']
    if file.filename == '':
        resp = jsonify({'message' : 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        df=pd.read_csv(BytesIO(file.read()))
        Customer_sentences=list(df[df['Speaker']=='Customer']['Sentence'])
        Agent_sentences=list(df[df['Speaker']=='Agent']['Sentence'])
        processed_df=processing(df)
        print(df.columns)
        summary=dict()
        summary['Outcome']='transferred/closed/redirected'
        summary['Overall_sentiment']='positive/negative/neutral'
#         summary["Outcome"]=status_of_the_call(processed_df,status_model_name)
#         summary['Overall_sentiment']=sentiment_of_the_call(processed_df,sentiment_model_name)
        r2=reason_for_the_call(Customer_sentences,reason_classification_model, reason_bart_model)
        r1=action_item(Agent_sentences,action_classification_model, action_bart_model)
        for i in range(len(r1)):
            p=r1[i].strip().split()[0]
            p1=r1[i].strip().split()
            if p.lower() != 'agent':
                if p.lower().endswith('ent'):
                    p1[0]='Agent'
                else:
                    p1.insert(0,'Agent')
            r1[i]=' '.join(p1)
        summary['Action_item']=r1
        
        for i in range(len(r2)):
            p=r2[i].strip().split()[0]
            p1=r2[i].strip().split()
            if p.lower() != 'customer':
                if p.lower().endswith('mer'):
                    p1[0]='Customer'
                else:
                    p1.insert(0,'Customer')
            r2[i]=' '.join(p1)
        summary['Reason']=r2
        
        #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        resp = jsonify(summary)
        resp.status_code = 200
        return resp
    else:
        resp = jsonify({'message' : 'Allowed file types are csv,'})
        resp.status_code = 400
        return resp



if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 9000, debug = True)



