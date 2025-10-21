
## Setup

1. **Clone the repo and create a virtual environment**
    ```
    git clone https://github.com/<your-username>/sentimental-analysis.git
    cd sentimental-analysis
    python3 -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

2. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```

3. **Download data**
    - Download Amazon per-category review data (like Video_Games_5.json.gz) from [UCSD Amazon Dataset](http://jmcauley.ucsd.edu/data/amazon/)
    - Place downloaded files in the `data/` folder

4. **Preprocess data**
    ```
    python preprocess.py
    ```

5. **Train the model**
    ```
    python train.py
    ```

6. **Evaluate**
    ```
    python evaluate.py
    ```

7. **Run dashboard**
    ```
    streamlit run dashboard.py
    ```

8. **Data and model versioning (optional)**
    ```
    dvc add data/Video_Games_5.json.gz
    git add data/Video_Games_5.json.gz.dvc .gitignore
    git commit -m "Add dataset with DVC tracking"
    dvc push
    ```

## Requirements

See `requirements.txt`. Main tools:

- Python >= 3.8
- pandas, numpy, scikit-learn
- transformers, torch, datasets
- streamlit, matplotlib, seaborn
- dvc, mlflow

## Usage

- View sentiment distribution and trends for your reviews
- Predict sentiment for any input review text in real time
- Track experiments and model improvements
- Extend for continuous retraining/MLOps workflows

## License

MIT

## Author

Your Name ([Anubhav Mandarwal](https://github.com/VrityaCodeRishi))
