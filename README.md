# End-to-End-deployment-of-a-Crop-recommendation-system
Deployed an Artificial Neural Network (ANN) on a Flask web app for crop recommendations. Utilized TensorFlow and Keras for model creation. The system considers factors like soil composition, weather, and pH. Users input parameters for real-time, accurate crop suggestions. Enhance agriculture decisions with this data-driven recommendation tool.


**Overview:**
This project employs ANNs to recommend crops based on input factors such as nitrogen, phosphorus, potassium, city, rainfall, pH, and real-time weather data. The ANN model is trained on a dataset, and predictions are served through a Flask web application with HTML and CSS.

**Detailed Overview:**

**1. Dataset and Preprocessing:**
   - Utilized a dataset containing agricultural data with features like nitrogen, phosphorus, and more.
   - Applied one-hot encoding to the dependent variable for multi-class classification.

**2. ANN Model:**
   - Constructed an ANN model using TensorFlow and Keras.
   - Configured layers with activation functions for optimal learning.
   - Compiled the model with categorical crossentropy loss for multi-class classification.

**3. Flask Web Application:**
   - Developed a Flask web application to provide a user-friendly interface.
   - Created an HTML form for users to input agricultural factors.
   - Integrated real-time weather data using the Weatherstack API.

**4. Deployment and Integration:**
   - Deployed the ANN model on the web using Flask, enabling real-time crop recommendations.
   - Saved the trained model, standard scalar, and column transformer using pickle.

**5. User Interface:**
   - Designed a clean and intuitive user interface with HTML and CSS.
   - Users can input agricultural parameters, and the system predicts the recommended crop.

**Usage:**
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the Flask application with `python app.py`.
4. Access the web application at `http://localhost:5000` in your browser.

**Contributions:**
Contributions and enhancements are welcome. Feel free to open issues, provide feedback, or submit pull requests.

**Tools and Libraries:**
- Python, TensorFlow, Flask, HTML, CSS, Weatherstack API

**Acknowledgments:**
Acknowledgments to the machine learning and agriculture communities for valuable insights.



# Important 
Another model using Gaussian Naive Bayes is also put in the code(commented out) and the related pickle files are also there feel free to use that if that suits you and your project more.
app.yaml is also there for Google Cloud Implementation.
