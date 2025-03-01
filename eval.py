from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from retrieve import get_format_relevant_documents
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge import Rouge
from api import vector_store




def load_random_samples(n=10):
    """Charge n exemples aléatoires du dataset"""
    df = pd.read_csv("./downloaded_files/medquad.csv")
    return df.sample(n=n, random_state=42)

def answer(question):

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        prompt = ChatPromptTemplate.from_messages(
            messages=[
                (
                    "system",
                    """You are MEDIA, a medical assistant. You are here to help people with their medical questions.
                    answer ploitely and provide the best information possible.
                    here are the relevant documents: {reference}
                    use the following information to answer the question: {question}
                    instructions:
                    0.use only the given refrence to answer the user question.
                    1.don't provide any medical advice.
                    2. don't provide any personal information.
                    3. don't provide any information that is not in the reference.
                    4. don't provide any information that is not related to the question.
                    5.if the question is not clear, ask the user to provide more information.
                    6.if the user ask for medical advice, tell them to consult a doctor.
                    7.if the user ask the question in a different language,answer the question in that language.
                    8.always tell the user to consult a doctor if they have any medical concerns.
                    9.always add to the answer that the information is from the reference and the focus area of the reference as provided in the reference.
                    

                    """,
                ),
                ("human", "The question is: {question}"),
            ]
        )

        chain = prompt | llm
        answer=chain.invoke({
                "question": question,
                "reference": get_format_relevant_documents(question, vector_store)
            }).content
        return  answer


# Download necessary NLTK data
nltk.download('punkt')

def evaluate_rag_system():
    """
    Evaluates the RAG system by comparing generated answers with ground truth answers
    using various metrics including relevance, coherence, and factual accuracy.
    """
    # Load sample data
    samples = load_random_samples(n=10)
    
    # Initialize metrics storage
    results = {
        'question': [],
        'ground_truth': [],
        'generated_answer': [],
        'cosine_similarity': [],
        'bleu_score': [],
        'rouge_score': [],
        'length_ratio': [],
    }
    
    # Initialize sentence transformer model for embedding generation
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    rouge = Rouge()
    
    print("Evaluating RAG system responses...")
    
    # Process each sample
    for _, row in tqdm(samples.iterrows(), total=len(samples)):
        question = row['question']
        ground_truth = row['answer']
        
        # Get RAG response
        try:
            generated_answer = answer(question)
            
            # Calculate embeddings
            gt_embedding = encoder.encode([ground_truth])[0]
            gen_embedding = encoder.encode([generated_answer])[0]
            
            # Calculate cosine similarity
            similarity = cosine_similarity([gt_embedding], [gen_embedding])[0][0]
            
            # Calculate BLEU score
            reference = [word_tokenize(ground_truth.lower())]
            candidate = word_tokenize(generated_answer.lower())
            bleu = sentence_bleu(reference, candidate)
            
            # Calculate ROUGE score
            try:
                rouge_scores = rouge.get_scores(generated_answer, ground_truth)[0]
                rouge_l_f = rouge_scores['rouge-l']['f']
            except:
                rouge_l_f = 0
            
            # Calculate length ratio (generated answer / ground truth)
            length_ratio = len(generated_answer) / max(1, len(ground_truth))
            
            # Store results
            results['question'].append(question)
            results['ground_truth'].append(ground_truth)
            results['generated_answer'].append(generated_answer)
            results['cosine_similarity'].append(similarity)
            results['bleu_score'].append(bleu)
            results['rouge_score'].append(rouge_l_f)
            results['length_ratio'].append(length_ratio)
            
        except Exception as e:
            print(f"Error processing question: {question}")
            print(f"Error: {str(e)}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate aggregate metrics
    avg_similarity = results_df['cosine_similarity'].mean()
    avg_bleu = results_df['bleu_score'].mean()
    avg_rouge = results_df['rouge_score'].mean()
    avg_length_ratio = results_df['length_ratio'].mean()
    
    print("\n===== RAG Evaluation Results =====")
    print(f"Average Cosine Similarity: {avg_similarity:.4f}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Average ROUGE-L F1 Score: {avg_rouge:.4f}")
    print(f"Average Length Ratio: {avg_length_ratio:.2f}")
    
    # Save detailed results
    results_df.to_csv("rag_evaluation_results.csv", index=False)
    print("Detailed results saved to 'rag_evaluation_results.csv'")
    
    return results_df

if __name__ == "__main__":
    evaluate_rag_system()