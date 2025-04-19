# Crypto Price Prediction Application

Welcome to the Crypto Price Prediction Application! This project utilizes machine learning techniques (e.g. TensorFlow and scikit-learn) to predict future cryptocurrency prices based on historical data and technical indicators.

# Getting Started

To get started with the Crypto Price Prediction Application (Version 2.0+), follow these steps:

1. **Download the Project Files**  
   Download the zip file of this repository and unzip it to your desired location on your computer.

2. **Install Required Python Packages**  
   Ensure you have Python 3.x installed on your system. Install the required Python packages using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

   This will install all the necessary dependencies for the backend (Flask API) to run properly.

3. **Install Frontend Dependencies**  
   Navigate to the `frontend` directory and install the required Node.js packages:

   ```bash
   cd frontend
   npm install
   ```

4. **Run the Flask Backend**  
   In the project root directory, start the Flask API server:

   ```bash
   python flask_api.py
   ```
   The backend will run at `http://localhost:5001` by default.

5. **Run the Next.js Frontend**  
   In a new terminal, from the `frontend` directory, start the frontend development server:

   ```bash
   npm run dev
   ```
   The frontend will run at `http://localhost:3000` by default.

6. **Access the Application**  
   Open your browser and go to `http://localhost:3000` to use the web interface. The frontend will communicate with the backend API for predictions and data.

# Update Log

## Version 2.0
- **Web-Based Architecture**: Migrated to a web application with a Flask API backend and a Next.js (React) frontend.
- **Modern UI**: Added a user-friendly web interface for predictions and data visualization.
- **API Endpoints**: Exposed endpoints for prediction, historical data, and comparison.

## Version 1.3
- **Added New Crypto**: Ethereum (ETH) and Dogecoin (DOGE)

## Version 1.2
- When the user chooses to view the historical data or compare historical and predicted data, they can select a time frame: one week, one month, one year, or all time.
  
## Version 1.1

- **Added Basic User Interface**: Implemented a command-line interface for easy interaction with the application.
- **Real-Time Data Support**: The application now fetches and uses the most recent cryptocurrency data for predictions.
