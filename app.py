import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

main_data = pd.read_csv('preprocessed.csv')
data = main_data.copy()
vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
vectorizer_matrix = vectorizer.fit_transform(data['soup'])

cos_sim = cosine_similarity(vectorizer_matrix,vectorizer_matrix)
data = data.reset_index()
title = data['title']
indices = pd.Series(data.index,index=title)


def final_reco(movi):
    movi = movi.lower()
    try:
        idx = indices[movi]
        sim_score = list(enumerate(cos_sim[idx]))
        sim_score = sorted(sim_score,key= lambda x: x[1],reverse=True)
        sim_score = sim_score[1:20]
        movie_ind = [i[0] for i in sim_score]
        movi = data.iloc[movie_ind][['title','vote_average','vote_count','year']]
        vote_count = movi[movi['vote_count'].notnull()]['vote_count'].astype('int')
        vote_ava = movi[movi['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_ava.mean()
        m = vote_count.quantile(0.60)
        quantifi = movi[(movi['vote_count'] >= m) & (movi['vote_count'].notnull()) & (movi['vote_average'].notnull())]
        quantifi['vote_count'] = quantifi['vote_count'].astype('int')
        quantifi['vote_average'] = quantifi['vote_average'].astype('int')
        def weig_rat(x):
            v = x['vote_count']
            R = x['vote_average']
            return (v / (v + m) * R) + (m / (m + v) * C)
        quantifi['wr'] =  quantifi.apply(weig_rat, axis=1)
        quantifi =quantifi.sort_values('wr', ascending=False).head(10)
        print(quantifi)
        return quantifi

    except Exception as e:
        print(e)
        return ''

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')
@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = final_reco(movie)

    movie = movie.upper()
    if type(r) == str:
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie = movie,r=r,t = 'l')

if __name__ == '__main__':
    app.run(debug=True)