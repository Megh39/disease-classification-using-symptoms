<h1 align="center" style="color:#2E8B57; font-size:36px;">Disease Prediction Using Symptoms</h1>
<p align="center" style="font-size: 18px;">A Machine Learning Project for Multi-class Disease Classification</p>

<hr />

<h2 style="color:#1E90FF;">Project Overview</h2>
<p style="font-size:17px;">
This project aims to build a machine learning system capable of predicting probable diseases based on binary symptom inputs. Each patient is represented by a vector of 377 binary features indicating the presence or absence of symptoms. The goal is to classify records into one of 678 disease categories after preprocessing, while handling real-world challenges such as extreme class imbalance and high sparsity.
</p>

<p style="font-size:16px;">
Key challenges addressed:
</p>
<ul style="font-size:16px;">
  <li>High number of classes with imbalanced distribution</li>
  <li>Over 90% sparsity across symptom features</li>
  <li>Efficient multi-class classification using linear models</li>
  <li>Evaluation based on balanced metrics for fairness</li>
</ul>

<hr />

<h2 style="color:#FF4500;">Dataset Details</h2>
<ul style="font-size:16px;">
  <li><b>Dataset Name:</b> Disease and Symptoms Dataset</li>
  <li><b>Source:</b> Mendeley Data</li>
  <li><b>Link:</b> <a href="https://data.mendeley.com/datasets/2cxccsxydc/1" target="_blank">https://data.mendeley.com/datasets/2cxccsxydc/1</a></li>
  <li><b>Total Records:</b> ~246,000 patients</li>
  <li><b>Features:</b> 377 symptoms (binary encoded)</li>
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
 │     └── final.ipynb     # EDA and model development
 ├── README.md             # Project documentation
 └── requirements.txt      
</pre>

<hr />

<h2 style="color:#2F4F4F;">Dependencies and Installation</h2>
<p style="font-size:16px;">To install all required libraries, run:</p>
<pre><code>pip install -r requirements.txt</code></pre>
<p style="font-size:16px;">Key libraries used:</p>
<ul style="font-size:16px;">
  <li>NumPy & Pandas – data processing</li>
  <li>Scikit-learn – machine learning models</li>
  <li>Imbalanced-learn – SMOTE and under sampling techniques</li>
  <li>Matplotlib & Seaborn – data visualization</li>
</ul>

<hr />

<h2 style="color:#DC143C;">Exploratory Data Analysis (EDA)</h2>
<p style="font-size:16px;">
A detailed Exploratory Data Analysis (EDA) was performed to understand the structure, patterns, and challenges within the dataset. Below are the key steps and findings:
</p>

<h3 style="color:#444;">1. Data Shape and Memory Usage</h3>
<ul>
  <li>~246k samples, 377 features, 1 target</li>
  <li>Data is moderate in size but sparse</li>
</ul>

<h3 style="color:#444;">2. Sparsity Check</h3>
<ul>
  <li>Over 90% of values in the symptom matrix are zero</li>
  <li>Average number of symptoms per patient: ~8 out of 377</li>
  <li>Created a heatmap to visualize sparsity structure</li>
</ul>

<h3 style="color:#444;">3. Disease Distribution</h3>
<ul>
  <li>Class imbalance detected: a few diseases dominate</li>
  <li>Plot: Bar chart of top 15 diseases by sample count</li>
  <li>Disease counts ranged from 1 to 1500+</li>
</ul>

<h3 style="color:#444;">4. Symptom Frequency</h3>
<ul>
  <li>Analyzed percentage occurrence for each symptom</li>
  <li>Identified top 10 most common symptoms</li>
</ul>

<h3 style="color:#444;">5. Handling Rare Classes</h3>
<ul>
  <li>Disease labels with fewer than 10 samples were grouped as <b>'others'</b></li>
  <li>This helps mitigate noise in model training</li>
</ul>

<h3 style="color:#444;">6. Visualization Highlights</h3>
<ul>
  <li>Sparsity heatmap</li>
  <li>Symptom occurrence distribution</li>
  <li>Disease frequency bar plot</li>
</ul>

<hr />

<h2 style="color:#CD5C5C;">Modeling Approach</h2>
<ol style="font-size:16px;">
  <li>Filtered and grouped disease labels based on occurrence frequency.</li>
  <li>Performed under-sampling to cap major classes to 500 samples.</li>
  <li>Applied SMOTE to oversample minor classes on the training set only.</li>
  <li>Trained three sparse-tolerant models (shown below).</li>
  <li>Evaluated using balanced metrics and Top-3 accuracy.</li>
</ol>

<hr />

<h2 style="color:#CD5C5C;">Models Used</h2>

<table style="width: 100%; border-collapse: collapse; font-size:16px;">
  <tr style="background:#f2f2f2;">
    <th>Model</th>
    <th>Why It Was Chosen</th>
    <th>Advantages</th>
  </tr>
  <tr>
    <td>Bernoulli Naive Bayes</td>
    <td>Efficient, handles binary data and high dimensionality well</td>
    <td>Fast, interpretable, ideal for sparse binary matrices</td>
  </tr>
  <tr style="background:#fafafa;">
    <td>Logistic Regression</td>
    <td>Suitable for multi-class problems, handles sparse data</td>
    <td>Supports large dimensionality and provides probabilities</td>
  </tr>
  <tr>
    <td>Linear SVC</td>
    <td>Effective for linearly separable data in high dimensions</td>
    <td>Scalable, robust, performs well on sparse data</td>
  </tr>
</table>

<hr />
<h2 style="color:#008080;">Evaluation Summary</h2>
<p style="font-size:16px;">
The models were evaluated using a combination of metrics designed specifically for multi-class classification in the presence of imbalanced data. In medical prediction tasks, where all classes are not equally represented, standard metrics like accuracy can be misleading. Hence, we used the following:</p>

<h3 style="color:#555;">Evaluation Metrics</h3>
<ul style="font-size:16px;">
  <li><b>Balanced Accuracy:</b> This metric adjusts for imbalanced class distributions by calculating the average recall across all classes. It ensures that diseases with fewer samples are fairly evaluated.</li>
  <li><b>Weighted F1 Score:</b> The harmonic mean of precision and recall, weighted by support (number of samples in each class). It gives more importance to common diseases but still accounts for less frequent ones.</li>
  <li><b>Top-3 Accuracy:</b> This measures whether the correct disease is among the model’s top 3 predictions. In real-world diagnosis scenarios, this is clinically valuable—the doctor can consider a few likely diseases before concluding.</li>
</ul>

<h3 style="color:#555;">Summary of Findings</h3>
<p style="font-size:16px;">
All three selected models—Bernoulli Naive Bayes, Logistic Regression, and Linear SVC—performed reasonably well, especially after applying data balancing techniques like undersampling and SMOTE. Logistic Regression and Linear SVC showed comparatively higher performance due to their compatibility with high-dimensional, sparse data. Bernoulli Naive Bayes, though simpler, demonstrated fast training time and served as a useful benchmark model.</p>

<p style="font-size:16px;">
Overall, the combination of oversampling, undersampling, and balanced evaluation metrics ensured that each model's performance was fairly assessed across all disease classes. This evaluation framework not only demonstrated model performance but also validated our data preprocessing and sampling strategy.</p>

<hr />

<h2 style="color:#2E8B57;">Future Work</h2>
<ul style="font-size:16px;">
  <li>Introduce deep learning like Transformers for sequence modeling</li>
  <li>Build a web UI or chatbot for interactive diagnosis prediction</li>
  <li>Apply explainable AI methods (SHAP/LIME) for model transparency</li>
  <li>Test with real-world electronic health record (EHR) data</li>
</ul>

<hr />

<h2 style="color:#2C3E50;">Acknowledgment</h2>
<ul style="font-size:16px;">
  <li>Dataset by Mendeley Data: Disease and Symptoms Dataset</li>
  <li>Libraries used: Scikit-Learn, NumPy, Pandas, Matplotlib, Imbalanced-learn, Seaborn</li>
  <li>Special thanks to authors and open-source contributors</li>
</ul>

<hr />

<h3 align="center" style="font-style:italic;">Thank you</h3>
