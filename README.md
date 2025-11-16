<h1 align="center" style="color:#2E8B57; font-size:36px;">Disease Prediction Using Symptoms</h1>
<p align="center" style="font-size: 18px;">A Machine Learning Project for Multi-class Disease Classification</p>

<hr />

<h2 style="color:#1E90FF;">Project Overview</h2>
<p style="font-size:17px;">
This project aims to build an automated machine learning system capable of predicting the most likely disease based on binary symptom inputs.  
Each patient is represented as a vector of 377 symptoms (0 = absent, 1 = present).  
The goal is to classify each patient into one of ~678 diseases after preprocessing, while addressing challenges such as:
</p>

<ul style="font-size:16px;">
  <li> Extremely imbalanced class frequencies</li>
  <li> 90% sparse input matrix</li>
  <li> Large number of disease classes</li>
  <li> Need for clinically relevant evaluation (Top-3 predictions)</li>
</ul>

<hr />

<h2 style="color:#FF4500;">Dataset Details</h2>
<ul style="font-size:16px;">
  <li><b>Dataset Name:</b> Disease and Symptoms Dataset</li>
  <li><b>Source:</b> Mendeley Data</li>
  <li><b>Link:</b> <a href="https://data.mendeley.com/datasets/2cxccsxydc/1" target="_blank">https://data.mendeley.com/datasets/2cxccsxydc/1</a></li>
  <li><b>Total Records:</b> ~246,000 patients</li>
  <li><b>Features:</b> 377 binary symptoms</li>
  <li><b>Targets:</b> 773 diseases (later grouped to ~678)</li>
</ul>

<hr />

<h2 style="color:#8A2BE2;">Team Members</h2>
<table style="width: 60%; font-size:16px; border-collapse: collapse;">
  <tr><th>Student ID</th><th>Name</th></tr>
  <tr><td>202518018</td><td>Megh Nanavati</td></tr>
  <tr style="background:#f9f9f9;"><td>202518021</td><td>Ambuj Tripathi</td></tr>
  <tr><td>202518022</td><td>Manya Choradiya</td></tr>
  <tr style="background:#f9f9f9;"><td>202518026</td><td>Pallavi Raghuvanshi</td></tr>
</table>

<hr />

<h2 style="color:#2F4F4F;">Project Structure</h2>
<pre style="background:#f4f4f4; padding:10px; border-radius:6px; font-size:15px;">
Disease-Prediction-ML/
 ├── data/
 │     └── Disease and Symptoms Dataset.csv
 ├── notebooks/
 │     └── final.ipynb          # EDA, training & inference
 ├── README.md                  # Documentation
 └── requirements.txt
</pre>

<hr />

<h2 style="color:#2F4F4F;">Dependencies and Installation</h2>
<p style="font-size:16px;">Install all required libraries:</p>

<pre><code>pip install -r requirements.txt</code></pre>

<b>Major libraries used:</b>
<ul style="font-size:16px;">
  <li>NumPy, Pandas – data processing</li>
  <li>Scikit-learn – ML models</li>
  <li>Imbalanced-learn – data balancing</li>
  <li>Matplotlib, Seaborn – visualization</li>
  <li>SentenceTransformer, RapidFuzz – NLP matching</li>
</ul>

<hr />

<h2 style="color:#DC143C;">Exploratory Data Analysis (EDA)</h2>
<p style="font-size:16px;">
A complete EDA was performed to understand dataset structure and challenges:
</p>

<h3 style="color:#444;">1. Dataset Shape & Sparsity</h3>
<ul>
  <li>246,945 rows × 377 binary symptoms</li>
  <li>Average active symptoms per patient: ~8</li>
  <li>Matrix sparsity: ~92%</li>
</ul>

<h3 style="color:#444;">2. Disease Frequency</h3>
<ul>
  <li>Some diseases have >1200 samples</li>
  <li>Many diseases have ≤5 samples → removed or grouped</li>
</ul>

<h3 style="color:#444;">3. Top Symptoms</h3>
<ul>
  <li>Symptoms like sharp abdominal pain, vomiting, headache appear most frequently</li>
</ul>

<h3 style="color:#444;">4. Visualization Outputs</h3>
<ul>
  <li>Disease frequency bar chart</li>
  <li>Symptom frequency bar chart</li>
  <li>Sparse heatmap showing symptom activation patterns</li>
</ul>

<hr />

<h2 style="color:#CD5C5C;">Modeling Approach</h2>
<ol style="font-size:16px;">
  <li>Grouped diseases below a minimum frequency threshold</li>
  <li>Under-sampled frequent classes (max 500 each)</li>
  <li>Trained 3 high-dimensional sparse-friendly models</li>
  <li>Combined them using a soft-voting ensemble</li>
  <li>Evaluated using clinically meaningful metrics</li>
</ol>

<hr />

<h2 style="color:#CD5C5C;">Models Used</h2>

<table style="width: 100%; border-collapse: collapse; font-size:16px;">
  <tr style="background:#f2f2f2;">
    <th>Model</th>
    <th>Why Selected</th>
    <th>Advantages</th>
  </tr>
  <tr>
    <td>Bernoulli Naive Bayes</td>
    <td>Matches binary sparse data assumptions</td>
    <td>Fast, memory efficient, interpretable baseline</td>
  </tr>
  <tr style="background:#fafafa;">
    <td>Logistic Regression</td>
    <td>Handles multi-class classification well</td>
    <td>Produces calibrated probabilities</td>
  </tr>
  <tr>
    <td>Linear SVC</td>
    <td>Performs extremely well in high-dimensional sparse space</td>
    <td>Strong margin-based classifier</td>
  </tr>
</table>

<hr />

<h2 style="color:#008080;">Final Ensemble Evaluation</h2>

<table style="width:60%; font-size:17px;">
<tr><td><b>Ensemble Accuracy</b></td><td>0.8704</td></tr>
<tr><td><b>Balanced Accuracy</b></td><td>0.8986</td></tr>
<tr><td><b>Macro F1 Score</b></td><td>0.8723</td></tr>
<tr><td><b>Weighted F1 Score</b></td><td>0.8712</td></tr>
<tr><td><b>Top-3 Accuracy</b></td><td>0.9633</td></tr>
</table>

<p style="font-size:16px;">
Top-3 accuracy is crucial in medical diagnosis since doctors consider differential diagnoses rather than a single label.
</p>

<hr />

<h2 style="color:#6A5ACD;">NLP Symptom Matching System</h2>

<p style="font-size:16px;">
A natural language interface was implemented so users can enter free-text queries instead of clicking symptoms manually.
</p>

<b>Pipeline:</b>

<ol style="font-size:16px;">
 <li>Sentence Transformer embeds all 377 symptoms</li>
 <li>User text is split and cleaned</li>
 <li>Cosine similarity finds semantically closest symptoms</li>
 <li>Fuzzy matching handles spelling mistakes</li>
 <li>Top symptoms converted to a binary vector</li>
 <li>Vector passed to the ML ensemble → disease prediction</li>
</ol>

<b>Example Input:</b>

<pre> "severe headache with vomiting" </pre>

<b>Matched Symptoms:</b>  
vomiting, headache, nausea

<b>Predicted Diseases:</b>
<pre>
1. Meningitis                     (0.120)
2. Headache after lumbar puncture (0.100)
3. Problem during pregnancy       (0.088)
</pre>

<hr />

<h2 style="color:#2E8B57;">Future Work</h2>
<ul style="font-size:16px;">
  <li>Transformer-based disease classifier</li>
  <li>Chatbot or web-based diagnosis UI</li>
  <li>Explainability integration (SHAP / LIME)</li>
  <li>Testing on real hospital EHR data</li>
</ul>

<hr />

<h2 style="color:#2C3E50;">Acknowledgment</h2>
<ul style="font-size:16px;">
  <li>Mendeley Dataset – Disease & Symptom Dataset</li>
  <li>Libraries used: Scikit-Learn, NumPy, Pandas, SentenceTransformer</li>
  <li>Thanks to open-source contributors</li>
</ul>

<hr />

<h3 align="center" style="font-style:italic;">Thank you</h3>
