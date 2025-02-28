# 🏥 MEDIA - Medical Chatbot with RAG & Cloud SQL 🚀  

AIHealthBot is an intelligent medical chatbot based on **LangChain**, **Vertex AI**, and **PostgreSQL**.  
It uses a **Retrieval-Augmented Generation (RAG)** system to provide accurate medical responses from a vector database.  

---

## ✨ Features  
- 🔍 **Vector Search**: Uses `textembedding-gecko@latest` to find the most relevant medical questions.  
- 📚 **Knowledge Base**: Integrated with **PostgreSQL + pgvector** to store embeddings and metadata.  
- ⚡ **FastAPI & Streamlit**: REST API for AI + user interface.  
- ☁ **Cloud Deployment**: Runs on **Google Cloud SQL & Vertex AI**.  

---  

## Project Structure 📚  

Here’s an overview of the main files and directories in the project:  

```plaintext
AIHealthBot/
├── downloaded_files/         # Contains the CSV file 'medquad.csv' downloaded from the GCS bucket  
├── .gitignore                # List of files ignored by Git  
├── api.py                    # FastAPI backend for interacting with AI  
├── app.py                    # User interface (Streamlit)  
├── config.py                 # Configuration for cloud variables  
├── eval.py                   # Evaluation script  
├── ingest.py                 # Data loading and indexing  
├── gcs_to_cloudsql.ipynb     # notebook for data transfer
│── README.md                 # Project documentation  
├── requirements.txt          # List of Python dependencies  
├── retrieve.py               # Retrieval and search for relevant documents  
```  

---  

## 🛠️ Installation & Configuration  

### 🔧 Initial Setup on Google Cloud  

Before running the application, make sure to configure **Google Cloud** by creating the following resources:  

- **Create a Google Cloud Storage (GCS) bucket**: The bucket will be used to **store CSV files** and other necessary resources.  
- **Create a Cloud SQL instance**  
- **Create the database linked to the instance**  

### 1️⃣ - **Clone the Project**  
```bash
git clone (https://github.com/yassineba13/MedIA.git)
cd MEDAI
```  

### 2 - **Create a Virtual Environment & Install Dependencies**  
```bash
python3 -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```  

### 3 - **Configure API Key & Cloud DB Password**  

- Create a `.env` file in the project root and add the API key and password:  
  ```ini
  GOOGLE_API_KEY=your_api_key
  DB_PASSWORD=yourpassword
  ```  
- Add `.env` to the `.gitignore` file to prevent committing the API key and password to Git.  

### 4 - **Configure Cloud Variables**  

- Create a `config.py` file in the project root and add the following Cloud variables:  
  ```ini
  PROJECT_ID=projectid
  INSTANCE=instancename
  REGION=yourregion
  DATABASE=yourdatabase
  DB_USER=postgresuser
  TABLE_NAME=yourtablename
  BUCKET_NAME=bucketname
  ```  

### 5 - **Run Cloud SQL Proxy**  
```bash
./cloud-sql-proxy --port port projectid:region:instancename
```  

### 6 - **Create PostgreSQL Table**  
The creation of the table containing embeddings and metadata for medical questions is included in the project notebook (`notebook.ipynb`).  

Run the notebook to automatically generate the table and insert the data.  

See the image below for the detailed structure of the SQL table:  

![table](images/table.png)  

---  

## 🚀 Usage  

### 1 - **Start the FastAPI Backend**  
The API is developed with **FastAPI** and allows interaction with AI through HTTP requests.  

```bash
uvicorn api:app --reload
```  
Make sure to define the HOST variable in `app.py`.  

### 2 - **Launch the User Interface (Streamlit)**  
```bash
streamlit run app.py
```  

---  

## 🐝 License  
This project is licensed under the **MIT License**.  

