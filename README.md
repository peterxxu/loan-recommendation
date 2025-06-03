# Loan Recommendation System

This project is a Streamlit web application designed to provide insights into historical Request for Proposals (RFPs) and recommend suitable lenders based on RFP characteristics and lender preferences. It uses a machine learning model (XGBoost) to predict the likelihood of successful funding for RFP-lender matches.

## Video Presentation

Watch a video presentation and demo of the project: [Loom Video](https://www.loom.com/share/f46d08c5a7de461182762562d0e5dccc?sid=fa9641d5-76d0-4a77-b3f7-faeb51cd0438)

## Features

- **Exploratory Data Analysis (EDA):** Visualizations and overviews of historical RFP data and lender preferences.
- **Model Insights:** Feature importances and SHAP value plots to understand the factors driving funding success.
- **Lender Recommendation:** An interactive form to input RFP details and receive a ranked list of potential lenders with predicted funding probabilities.
- **Methodology Overview:** Explanation of the approach, model used, and potential future improvements.

## Development Approach and Visualizations

This project includes a Jupyter Notebook, `pipeline.ipynb`, which serves as the primary environment for detailed development, data processing, model training, and generation of analytical insights, including visualizations.

The Streamlit application (`app.py`) is primarily a demonstration tool that showcases the results and the recommendation engine. Many of the visualizations displayed in the Streamlit app are pre-computed and saved as static image files (e.g., in an `assets/` directory) generated from the `pipeline.ipynb` notebook, rather than being computed in real-time within the app itself. This approach allows for a more focused and performant demonstration in the Streamlit app.

## Data

The application requires two CSV files located in a `data/` subdirectory:

- `data/historical_rfps.csv`: Contains historical data of RFPs.
- `data/lender_preferences.csv`: Contains information about lender preferences.

Ensure these files are present in the specified location before running the app. Sample data or instructions on how to obtain/generate this data would typically be included here.

## Setup and Installation

1.  **Clone the repository (if applicable):**

    ```bash
    # git clone <repository-url>
    # cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    xgboost
    scikit-learn
    shap
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Once the setup is complete and the data files are in place, run the Streamlit application using the following command in your terminal:

```bash
streamlit run app.py
```

This will typically open the application in your default web browser.
