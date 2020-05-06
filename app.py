from flask import Flask, request, render_template, jsonify
import nltk
from autocorrect import spell
from gensim.summarization import summarize as g_sumn

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/keyword', methods=["GET"])
def keyword_extraction():
    return render_template('keyword.html')


@app.route('/remove_numbers', methods=["GET"])
def others():
    return render_template('remove_numbers.html')


@app.route('/lowercase', methods=["GET"])
def lower():
    return render_template('lowercase.html')


@app.route('/home', methods=["GET", "POST"])
def lower_case():
    text1 = request.form['text']
    word = text1.lower()
    result = {
        "result": word
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


@app.route('/lowercase', methods=["GET", "POST"])
def lower_case_home():
    text1 = request.form['text']
    word = text1.lower()
    result = {
        "result": word
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


@app.route('/remove_numbers', methods=["GET", "POST"])
def remove_numbers():
    text = request.form['text']
    remove_num = ''.join(c for c in text if not c.isdigit())
    result = {
        "result": remove_num
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


@app.route("/keyword", methods=["GET", "POST"])
def keyword():
    text = request.form['text']
    word = nltk.word_tokenize(text)
    pos_tag = nltk.pos_tag(word)
    chunk = nltk.ne_chunk(pos_tag)
    NE = [" ".join(w for w, t in ele) for ele in chunk if isinstance(ele, nltk.Tree)]
    result = {
        "result": NE
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)


if __name__ == '__main__':
    app.run(debug=True)
