from flask import Flask
from flask import request, render_template
import model

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def start_page():
    if request.method == 'POST':
        context = request.form['context']
        question = request.form['question']
        result = model.infer(context=context, question=question)
        return render_template('start_with_answer.html',
                               name=None,
                               context=context,
                               question=question,
                               answer=result)
    else:
        return render_template('start.html', name=None)
