from flask import Flask, render_template, request, redirect
from datetime import datetime

app=Flask(__name__)

tasks=[]

class Task:
    c=0
    def __init__(self,name):
        self.name=name
        self.date=datetime.now().time()
        self.id=Task.c
        Task.c+=1
        print(self.id)



@app.route('/')
def run():
    return render_template('index.html',tasks=tasks)

@app.route('/create',methods=['GET','POST'])
def create():
    if request.method=='GET':
        return render_template('create.html')
    if request.method=='POST':
        text=request.form['content']
        new_task=Task(text)
        tasks.append(new_task)
        return redirect('/')
    
@app.route('/delete/<int:id>')
def delete(id):
    for ind,task in enumerate(tasks):
        if(task.id==id):
            tasks.pop(ind)
            break
    return redirect('/')
    
if __name__ == "__main__":
    app.run(debug=True)