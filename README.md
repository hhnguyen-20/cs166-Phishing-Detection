
# CS166-Phishing-Detection

## Project Introduction:

Welcome to our Email Phishing Detection project! This project utilizes machine learning techniques to classify emails as either spam or legitimate based on their content, specifically the email title and body text. By training models on labeled email data, we've developed a robust system capable of accurately discerning between phishing attempts and genuine communications.

## Getting Started:

To get started with our project, follow these simple steps to set up your development environment:

1. **Clone the Repository:**
   ```
   git clone https://github.com/your-username/email-phishing-detection.git
   cd email-phishing-detection
   ```

2. **Install Dependencies:**
   Create a new Python environment and install the required packages using pip:
   ```
   pip install flask pdfplumber beautifulsoup4 joblib pandas matplotlib numpy seaborn scikit-learn nltk
   ```

3. **Running the Full Stack App:**
   After setting up the environment, navigate to the 'support_vector_machine' directory and execute the 'app.py' file. Make sure to run all necessary Python files to train and save the model before running the app.

   ```
   cd Support\ Vector\ Machine
   python app.py
   ```

### Email Scraping:
In the 'email_scrape' directory, you'll find a 'test.py' file that implements the Gmail API to obtain emails. These emails can be classified using the trained model to determine whether they are spam or legitimate. Before using this feature, ensure you have trained the necessary models, particularly for logistic regression, which is utilized by the API.

To allow the Gmail API to interact with your email, you need to set up your Google Cloud console. Sign in to Google Cloud console and create a new project. Within your dashboard go to API's and services and enable specifically the Gmail API. Next go to the configure consent screen and create your application with your application name. Next go to credentials and create an OAuth Client ID, select Desktop application, enter Application name, and click the create button. With the client ID created, down it to your computer as credentials.json. Now the code for the test.py file should work!

## Additional Sections:

### Datasets:
In the 'support_vector_machine' directory, you'll find a 'datasets' folder containing the dataset used to train the models. Feel free to explore them if you're interested in understanding the data used for training.

