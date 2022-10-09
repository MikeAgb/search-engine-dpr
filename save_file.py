from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from flask import Markup, send_from_directory

########## constants ########
PASSAGE_SIZE = 100
OVERLAP = 20
k = 2

######### models ###########
doc_encoder = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3')
query_encoder = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3')
vectorizer = TfidfVectorizer()

######### storage ##########
embedded_docs = []
name_to_embed = {}
all_passages = []
tfidf_matrices = []
all_passages_str = []

passages_to_article = {}


########### flask session ##########
app = Flask(__name__)
app.secret_key = "secret" # for encrypting the session

################### upload ################
path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def upload_form():
    return render_template('frontend.html')


###################### check file type #####################
ALLOWED_EXTENSIONS = set(['txt'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
####################################################


#################### get passages #######################
def get_passages(parsed_text):
    '''
    function to get passages with some overlap, making sure not to cut mid sentence
    TODO add other file types, and other punctuation types
    '''
    passages = []
    current_end = 0

    while current_end < len(parsed_text):
        current_passage = parsed_text[current_end: min(current_end + PASSAGE_SIZE, len(parsed_text)-1)]
        current_end += PASSAGE_SIZE-OVERLAP
        where_to_stop = current_end+OVERLAP
        if current_end >= len(parsed_text):
            break

        while where_to_stop  < len(parsed_text)-1 and parsed_text[where_to_stop][-1]!='.':
            where_to_stop +=1
            current_passage.append(parsed_text[where_to_stop])

        while current_end < len(parsed_text)-1 and parsed_text[current_end][-1]!='.':
            current_end+=1
        current_end+=1
        passages.append(current_passage[:])
    return passages

##########################################################


################### load docs into memory #################
@app.route('/uploader', methods = ['POST']) 
def parse_doc():

    global tfidf_matrices
    global all_passages_str

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No file selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        ### parse it into passages
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        fileObject = open(filename, "r")
        file = fileObject.read()
        parsed_text = file.split()
        passages = get_passages(parsed_text)
  
        #transformer
        embeddings = doc_encoder.encode(passages)
        embedded_docs.extend(embeddings)
        all_passages.extend(passages)

        #tfidf
        for p in all_passages:
            all_passages_str.append(' '.join(p))
            passages_to_article[' '.join(p)] =  filename


        flash('File successfully uploaded')
        return redirect('/')
    else:
        flash('Allowed file types are txt')
        return redirect('/')


################# parse query using transformer ############
@app.route('/query', methods=['POST'])
def parse_query():
    flash('checking databank')
    query = request.form['text']
    query_embedding = query_encoder.encode(query)
    scores = util.dot_score(query_embedding,embedded_docs)
    indexes = np.argsort(-scores[0])[0:k]

    i = 1
    for idx in indexes:
        flash(i)
        i+=1
        flash(' '.join(all_passages[idx]))
        filename = passages_to_article[all_passages_str[idx]]
        #flash('file name: ' + filename)

    return redirect('/')


############# parse query using tfidf #############
@app.route('/tfidf', methods = ['POST']) 
def tfidf():
    global all_passages_str

    flash('checking databank')
    query = request.form['text']

    all_passages_str.append(' '.join(query))
    tfidf = vectorizer.fit_transform(all_passages_str)
    similarity_matrix = cosine_similarity(tfidf)
    new_query_similarities = similarity_matrix[0]
    indexes = np.argsort(-new_query_similarities)[1:k+1]

    i = 1
    for idx in indexes:
        flash(str(i))
        i+=1
        flash(' '.join(all_passages[idx]))
    return redirect('/')


@app.route('/set_k', methods = ['POST']) 
def set_k():
    global k
    k = int(request.form['k'])
    flash('now returning: ' + str(k)+ ' queries')
    return redirect('/')


if __name__ == "__main__":
    app.run(host="127.0.0.1",port = 5000)

