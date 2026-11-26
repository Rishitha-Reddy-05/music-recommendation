import streamlit as st,numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.transform import resize
st.title("Handwritten Digit Recognition"); st.caption("Demo using scikit-learn's small 8x8 digits dataset.")
d=load_digits(); Xtr,Xte,ytr,yte=train_test_split(d.data,d.target,test_size=.2,random_state=42,stratify=d.target); m=RandomForestClassifier(n_estimators=150,random_state=42); m.fit(Xtr,ytr)
st.metric("Held-out test accuracy",f"{accuracy_score(yte,m.predict(Xte)):.1%}")
up=st.file_uploader("Upload a digit image (0-9)",type=["png","jpg","jpeg"])
if up:
 img=Image.open(up).convert("L"); a=np.array(img); sm=resize(a,(8,8),anti_aliasing=True)
 if sm.mean()>.5: sm=1-sm
 st.image(img,width=220); st.metric("Prediction",int(m.predict((sm*16).reshape(1,-1))[0])); st.caption("Uploaded-image prediction may be unreliable due to preprocessing differences.")
st.write("Dataset examples"); st.image(d.images[:20],caption=[str(x) for x in d.target[:20]],width=100)
