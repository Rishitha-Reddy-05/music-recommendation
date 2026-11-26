import streamlit as st,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
songs=[("Golden Hour","JVKE","pop romantic piano"),("Blinding Lights","The Weeknd","pop synth upbeat"),("Levitating","Dua Lipa","pop dance upbeat"),("Someone Like You","Adele","pop ballad emotional piano"),("Perfect","Ed Sheeran","pop romantic acoustic"),("Believer","Imagine Dragons","rock energetic"),("Thunder","Imagine Dragons","rock pop energetic"),("Lovely","Billie Eilish","pop emotional soft"),("Sunflower","Post Malone","pop hiphop relaxed"),("Counting Stars","OneRepublic","pop energetic"),("Night Changes","One Direction","pop acoustic emotional"),("Dance Monkey","Tones and I","pop dance upbeat")]
df=pd.DataFrame(songs,columns=["title","artist","tags"]); v=TfidfVectorizer(); mat=v.fit_transform(df.tags); sim=cosine_similarity(mat)
st.title("Music Recommendation Demo"); st.caption("Recommendations from sample song tags.")
sel=st.selectbox("Choose a song",df.title); i=df.index[df.title==sel][0]; ranked=sorted(enumerate(sim[i]),key=lambda x:x[1],reverse=True); ids=[j for j,s in ranked if j!=i][:5]
st.subheader("Similar songs"); st.dataframe(df.iloc[ids][["title","artist","tags"]]); st.subheader("Catalog"); st.dataframe(df)
