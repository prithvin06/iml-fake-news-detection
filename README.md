# Fake News Detection Using Logistic Regression

This project aims to build a machine learning model using Logistic Regression to detect whether a given news article is real or fake. The model is trained on a dataset of news articles labeled as real or fake.

## Dataset

The dataset used in this project is the "WELFake_Dataset.csv" file, which contains the following columns:

- `Unnamed: 0`: A unique identifier for each row.
- `title`: The title of the news article.
- `text`: The content of the news article.
- `label`: A binary label indicating whether the news article is real (1) or fake (0).

## Dependencies

The following Python libraries are required to run this project:

- `numpy`
- `pandas`
- `scikit-learn`

## Project Structure

The project consists of a single Jupyter Notebook file named `Fake_News_Detection_.ipynb`. The notebook contains the following steps:

1. **Importing Dependencies**: The required Python libraries are imported.
2. **Data Collection and Pre-Processing**: The dataset is loaded from the CSV file, and null values are replaced with an empty string.
3. **Label Encoding**: The string labels 'fake' and 'real' are encoded as 0 and 1, respectively.
4. **Data Splitting**: The dataset is split into training and testing sets.
5. **Feature Extraction**: The text data is transformed into numerical feature vectors using TF-IDF vectorization.
6. **Model Training**: A Logistic Regression model is trained on the training data.
7. **Model Evaluation**: The trained model's performance is evaluated on both the training and testing data using accuracy, confusion matrix, and classification report metrics.
8. **Building a Predictive System**: A simple predictive system is implemented to classify a given news article as real or fake.

## Usage

1. Clone the repository or download the `Fake_News_Detection_.ipynb` file.
2. Install the required dependencies.
3. Open the Jupyter Notebook file.
4. Run the cells in the notebook sequentially.

Note: The dataset file (`WELFake_Dataset.csv`) should be placed in the `/content/drive/MyDrive/Datasets/` directory for the notebook to run correctly. If you're running the notebook locally, update the file path accordingly.

## Results

The Logistic Regression model achieved an accuracy of approximately 95.86% on the training data and 94.42% on the test data. The classification report and confusion matrix are also provided in the notebook to evaluate the model's performance in more detail.

## Future Work

- Experiment with other machine learning models, such as Naive Bayes, Support Vector Machines, or deep learning approaches.
- Explore advanced text preprocessing techniques, like stemming, lemmatization, or n-gram feature extraction.
- Investigate the use of additional features, such as the news source or author information, to improve the model's performance.
- Develop a user-friendly interface for the predictive system, allowing users to input news articles and obtain real/fake predictions.

## License

This project is licensed under the [MIT License](LICENSE).# iml-fake-news-detection
