from flask import Flask, request, jsonify
from flask_cors import CORS
from chemNEETData import chem_NEET_data
from phyNEETData import phy_NEET_data
from bioNEET import bio_data
from phyData import phy_data
from chemData import chem_data
from methsData import MathsData
from highLight import highLight
from basetoimage import main
from textextract import TextExtractor, OS
from PIL import Image
from pytesseract import pytesseract
import base64
import io
import os
import json
import pandas as pd
import numpy as np
import fitz
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def fix_base64_padding(base64_string: str) -> str:
    """Add padding to the base64 string if necessary."""
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += '=' * (4 - missing_padding)
    return base64_string

def decode_base64_to_image(base64_string: str, output_path: str) -> None:
    """Decode a base64 string and save it as an image file."""
    try:
        # Remove the data URI scheme part if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Fix base64 padding
        base64_string = fix_base64_padding(base64_string)
        
        # Print the cleaned base64 string for debugging
        print(f"Base64 string after removing prefix: {base64_string[:30]}...")

        # Decode the base64 string
        image_data = base64.b64decode(base64_string)
        
        # Convert binary data to an image
        image = Image.open(io.BytesIO(image_data))
        
        # Save the image to a file
        image.save(output_path)
        print(f"Image saved successfully at {output_path}")
    except Exception as e:
        print(f"Error decoding or saving image: {e}")

def main(base64_string: str) -> str:
    # Path where the decoded image will be saved
    image_path = 's1.png'
    
    # Decode the base64 string and save the image
    decode_base64_to_image(base64_string, image_path)
    
    # Check if the image was saved correctly
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return ""
    
    # Create an instance of TextExtractor
    extractor = TextExtractor(os_type=OS.Window)
   
    # Extract text from the image
    try:
        extracted_text = extractor.extract_text(image_path)
        if not isinstance(extracted_text, str):
            extracted_text = str(extracted_text)
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""
    
    # Optionally, clean up the image file after processing
    if os.path.exists(image_path):
        os.remove(image_path)
    
    return extracted_text

def bio_data(file , paragraph):
   

    # Load the CSV file
    df = pd.read_csv(file)
    
    # Extract necessary columns
    questions = df['question']
    images = df['image']
    answers = df['ans']
  

    # Preprocess the input text and questions
    def preprocess(text):
        tokens = nltk.word_tokenize(text)
        return ' '.join(tokens)

    preprocessed_questions = [preprocess(str(question)) for question in questions]
    preprocessed_paragraph = preprocess(paragraph)

    # Calculate TF-IDF vectors and cosine similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(preprocessed_questions + [preprocessed_paragraph])
    similarity_matrix = cosine_similarity(vectors[-1], vectors[:-1])

    # Sort and filter the results based on the similarity scores
    threshold = 0.2 
    sorted_indices = similarity_matrix[0].argsort()[::-1]
    sorted_questions = [(i, questions[i], similarity_matrix[0][i]) for i in sorted_indices]

    top_n = 10
    response = {}
    suggestions = []
    
    # Prepare the response with the top N suggestions including question, image, and answer
    for i, (index, question, score) in enumerate(sorted_questions[:top_n], 1):
        if score > threshold:
            suggestions.append({
                "question": question,
                "ans": answers[index],
                "image": images[index]
            })

    # Return the suggestions or a message if no relevant data is found
    if suggestions:
        response['suggestions'] = suggestions
    else:
        response['message'] = "No related data found"

    return jsonify(response)

def chem_data(file , paragraph):
    print(paragraph)

    # Load the CSV file
    df = pd.read_csv(file)
    
    # Extract necessary columns
    questions = df['question']
    images = df['image']
    answers = df['ans']
    

    # Preprocess the input text and questions
    def preprocess(text):
        tokens = nltk.word_tokenize(text)
        return ' '.join(tokens)

    preprocessed_questions = [preprocess(str(question)) for question in questions]
    preprocessed_paragraph = preprocess(paragraph)

    # Calculate TF-IDF vectors and cosine similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(preprocessed_questions + [preprocessed_paragraph])
    similarity_matrix = cosine_similarity(vectors[-1], vectors[:-1])

    # Sort and filter the results based on the similarity scores
    threshold = 0.2 
    sorted_indices = similarity_matrix[0].argsort()[::-1]
    sorted_questions = [(i, questions[i], similarity_matrix[0][i]) for i in sorted_indices]

    top_n = 10
    response = {}
    suggestions = []
    
    # Prepare the response with the top N suggestions including question, image, and answer
    for i, (index, question, score) in enumerate(sorted_questions[:top_n], 1):
        if score > threshold:
            suggestions.append({
                "question": question,
                "score": score,
                "ans": answers[index],
                "image": images[index]
            })

    # Return the suggestions or a message if no relevant data is found
    if suggestions:
        response['suggestions'] = suggestions
    else:
        response['message'] = "No related data found"

    return jsonify(response)

def chem_NEET_data(file , paragraph):
 

    # Load the CSV file
    df = pd.read_csv(file)
    
    # Extract necessary columns()
    questions = df['question'].tolist()
    images = df['image'].tolist()
    answers = df['ans'].tolist()
   
    # Preprocess the input text and questions
    def preprocess(text):
        tokens = nltk.word_tokenize(text)
        return ' '.join(tokens)

    preprocessed_questions = [preprocess(str(question)) for question in questions]
    preprocessed_paragraph = preprocess(paragraph)

    # Calculate TF-IDF vectors and cosine similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(preprocessed_questions + [preprocessed_paragraph])
    similarity_matrix = cosine_similarity(vectors[-1], vectors[:-1])

    # Sort and filter the results based on the similarity scores
    threshold = 0.2 
    sorted_indices = similarity_matrix[0].argsort()[::-1]
    sorted_questions = [(i, questions[i], similarity_matrix[0][i]) for i in sorted_indices]

    top_n = 10
    response = {}
    suggestions = []
    
    # Prepare the response with the top N suggestions including question, image, and answer
    for i, (index, question, score) in enumerate(sorted_questions[:top_n], 1):
        if score > threshold:
            suggestions.append({
                "question": question,
                "score": score,
                "ans": answers[index],
                "image": images[index]
            })

    # Return the suggestions or a message if no relevant data is found
    if suggestions:
        response['suggestions'] = suggestions
    else:
        response['message'] = "No related data found"

    return jsonify(response)

def highLight(file , page_number):
 
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.pdf'):
        # Open the PDF file
        doc = fitz.open(stream=file.read(), filetype="pdf")

        # Check if the requested page number is valid
        if page_number < 0 or page_number >= len(doc):
            return jsonify({'error': 'Invalid page number'}), 400

        # Extract the text from the specified page
        page = doc[page_number-1]
        extracted_text = page.get_text()
        summarized_text = summarize_paragraph(extracted_text)

        # Highlight the summarized text on the PDF page and get the base64 image
        highlighted_image = image_extractor(summarized_text, page)

        return jsonify({
            'message': 'File uploaded and processed successfully',
            'highlighted_image': highlighted_image
        }), 200

    return jsonify({'error': 'Invalid file type'}), 400


def normalize_color(color):
    return tuple(c / 255.0 for c in color)
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def image_to_base64_converter(image_path):
    """Convert an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def image_extractor(array_of_text_to_highlight, page):
    # Highlight color: Purple (R: 128, G: 0, B: 128)
    highlight_color = normalize_color((255, 128, 64))

    for text_to_highlight in array_of_text_to_highlight:
        text_instances = page.search_for(text_to_highlight)

        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)
            highlight.set_colors(stroke=highlight_color)
            highlight.update()

    # Generate a pixmap of the page after highlighting
    pix = page.get_pixmap()

    # Save the image to a temporary file
    output_image_path = os.path.join(UPLOAD_FOLDER, 'highlighted_page.png')
    pix.save(output_image_path)

    # Convert the saved image to a base64 string
    base64_image = image_to_base64_converter(output_image_path)
    
    # Clean up the temporary image file
    os.remove(output_image_path)

    return base64_image
keywords = [
    "is defined as","it depends on","perpendicular", "refers to", "can be described as", "known as", 
    "means", "is characterized by", "is identified by", "property", 
    "concept", "principle","named reaction","causes", "phenomenon", "law", "rule", 
    "theory", "axiom", "postulate", "term", "equals", 
    "is equivalent to", "expression", "equation", "formula", 
    "constant", "variable", "coefficient", "factor", "derivative", 
    "integral", "product", "sum", "difference", "quotient", 
    "function", "inverse", "ratio", "proportion", "relation", 
    "square root", "cube root", "logarithm", "exponent", "power"
]
def summarize_paragraph(paragraph):
    # Tokenize the paragraph into sentences
    sentences = sent_tokenize(paragraph)
    
    # Define stop words
    stop_words = set(stopwords.words('english'))
    
    # Create a set of keywords for quick lookup
    keyword_set = set(keyword.lower() for keyword in keywords)
    
    # Tokenize and clean sentences
    cleaned_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        cleaned_sentences.append(filtered_words)
    
    # Extract sentences that contain any of the keywords
    relevant_sentences = [
        sentence for sentence, words in zip(sentences, cleaned_sentences)
        if keyword_set.intersection(words)
    ]
    print((relevant_sentences))
    return relevant_sentences


def check_paragraph_similarity(paragraph, df):

    vectorizer = TfidfVectorizer()

    documents = [paragraph] + df['Tokens'].tolist()

    tfidf_matrix = vectorizer.fit_transform(documents)

    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    most_similar_index = cosine_similarities.argmax()
    most_similar_score = cosine_similarities[0][most_similar_index]

    most_similar_chapter = df.iloc[most_similar_index]['chapter']
    
    return most_similar_chapter, most_similar_score

def MathsData(paragraph, TotalMathsMergedData, quetionsFile, formOfDocument):
    if formOfDocument == 0:
        df = pd.read_csv(TotalMathsMergedData)
        df['Tokens'] = df['Tokens'].astype(str)
        most_similar_chapter, similarity_score = check_paragraph_similarity(paragraph, df)

        if similarity_score > 0.5:
            print(f"The paragraph is similar to {most_similar_chapter} with a similarity score of {similarity_score:.2f}.")

            dframe = pd.read_csv(quetionsFile)

            print("Most similar chapter:", most_similar_chapter)
            print(dframe)
            chapter_allQuestions = dframe[dframe['chapter'] == most_similar_chapter]
            # print(chapter_allQuestions)
            quetion_Image = chapter_allQuestions['questionImage'].tolist()
            ans = chapter_allQuestions['ans'].tolist()

            

            # Convert the arrays to lists and return them
            result = {"quetion_Image": quetion_Image, "ans": ans}
            return result
        else:
            print("The paragraph is not similar to any chapter.")
            return "invalid Text"
    else:
        df = pd.read_csv(TotalMathsMergedData)
        df['Tokens'] = df['Tokens'].astype(str)
        # function call of image to text
        ExtractedText = ""  # Replace with actual extracted text
        most_similar_chapter, similarity_score = check_paragraph_similarity(ExtractedText, df)

        if similarity_score > 0.5:
            print(f"The paragraph is similar to Chapter {most_similar_chapter} with a similarity score of {similarity_score:.2f}.")
            dframe = pd.read_csv(quetionsFile)
            chapter_allQuestions = dframe[dframe['chapter'] == most_similar_chapter]
            quetion_Image = chapter_allQuestions['questionImage'].tolist()
            ans = chapter_allQuestions['ans'].tolist()

            # Convert the arrays to lists and return them
            result = {"quetion_Image": quetion_Image, "ans": ans}
            return result
        else:
            print("The paragraph is not similar to any chapter.")
            return "invalid Text"

def phy_data(file,paragraph):
   

    # Load the CSV file
    df = pd.read_csv(file)
    
    # Extract necessary columns
    questions = df['question']
    images = df['image']
    answers = df['ans']
   

    # Preprocess the input text and questions
    def preprocess(text):
        tokens = nltk.word_tokenize(text)
        return ' '.join(tokens)

    preprocessed_questions = [preprocess(str(question)) for question in questions]
    preprocessed_paragraph = preprocess(paragraph)

    # Calculate TF-IDF vectors and cosine similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(preprocessed_questions + [preprocessed_paragraph])
    similarity_matrix = cosine_similarity(vectors[-1], vectors[:-1])

    # Sort and filter the results based on the similarity scores
    threshold = 0.2 
    sorted_indices = similarity_matrix[0].argsort()[::-1]
    sorted_questions = [(i, questions[i], similarity_matrix[0][i]) for i in sorted_indices]

    top_n = 10
    response = {}
    suggestions = []
    
    # Prepare the response with the top N suggestions including question, image, and answer
    for i, (index, question, score) in enumerate(sorted_questions[:top_n], 1):
        if score > threshold:
            suggestions.append({
                "question": question,
                "score": score,
                "ans": answers[index],
                "image": images[index]
            })

    # Return the suggestions or a message if no relevant data is found
    if suggestions:
        response['suggestions'] = suggestions
    else:
        response['message'] = "No related data found"

    return jsonify(response)

def phy_NEET_data(file, paragraph):

    # Load the CSV file
    df = pd.read_csv(file)
    
    # Extract necessary columns
    questions = df['question']
    images = df['image']
    answers = df['ans']
  

    # Preprocess the input text and questions
    def preprocess(text):
        tokens = nltk.word_tokenize(text)
        return ' '.join(tokens)

    preprocessed_questions = [preprocess(str(question)) for question in questions]
    preprocessed_paragraph = preprocess(paragraph)

    # Calculate TF-IDF vectors and cosine similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(preprocessed_questions + [preprocessed_paragraph])
    similarity_matrix = cosine_similarity(vectors[-1], vectors[:-1])

    # Sort and filter the results based on the similarity scores
    threshold = 0.2 
    sorted_indices = similarity_matrix[0].argsort()[::-1]
    sorted_questions = [(i, questions[i], similarity_matrix[0][i]) for i in sorted_indices]

    top_n = 10
    response = {}
    suggestions = []
    
    # Prepare the response with the top N suggestions including question, image, and answer
    for i, (index, question, score) in enumerate(sorted_questions[:top_n], 1):
        if score > threshold:
            suggestions.append({
                "question": question,
                "score": score,
                "ans": answers[index],
                "image": images[index]
            })

    # Return the suggestions or a message if no relevant data is found
    if suggestions:
        response['suggestions'] = suggestions
    else:
        response['message'] = "No related data found"

    return jsonify(response)

class OS(enum.Enum):
    Mac = "Mac"
    Window = "Window"

class Lang(enum.Enum):
    ENG = "eng"
    # Add more languages as needed

class TextExtractor:
    def __init__(self, os_type: OS):
        print(f"Running on {os_type.value}")
        
        if os_type == OS.Window:
            print(1)
            windows_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(windows_path):
                print(2)
                pytesseract.tesseract_cmd = windows_path
            else:
                raise FileNotFoundError(f"Tesseract not found at {windows_path}. Please install it and provide the correct path.")
        
        elif os_type == OS.Mac:
            print("Assuming Tesseract is installed in the PATH environment variable on Mac.")
            # Add Mac-specific configuration if needed

    def extract_text(self, image_path: str) -> str:
        try:
            img = Image.open(image_path)
            extracted_text = pytesseract.image_to_string(img, lang=Lang.ENG.value)
            return extracted_text
        except Exception as e:
            return f"Error occurred: {e}"

app = Flask(__name__)

import pandas as pd
CORS(app=app)
@app.route('/get-example', methods=['GET'])
def get_example():
    data = {
        'message': 'This is a GET request',
        'status': 'success'
    }
    return jsonify(data)
  
import pandas as pd
import random

def get_random_questions(exam_type):
    """Fetch random questions from the respective CSV files based on the exam type."""
    question_files = []
    
    try:
        if exam_type == 'jee':
 
            chem_df = pd.read_csv(r'csvFiles\Jee_chem.csv')
            phy_df = pd.read_csv(r'csvFiles\Jee_physics.csv')
            math_df = pd.read_csv(r'csvFiles\Final_Maths_Jee.csv')
            question_files = [chem_df, phy_df, math_df]
        elif exam_type == 'neet':

            bio_df = pd.read_csv(r'csvFiles\Neet_BIo.csv')
            phy_df1 = pd.read_csv(r'csvFiles\Neet_phy.csv')
            chem_df1 = pd.read_csv(r'csvFiles\Neet_chem.csv')
            question_files = [bio_df, phy_df1, chem_df1]
        else:
            return None, None

        questions = []
        solutions = []
        for df in question_files:
            if df.empty:
                print("One of the DataFrames is empty.")
                continue

            try:
                
                selected_row = df.sample(1).iloc[0]
                if exam_type == 'jee':
                    if df.equals(math_df):
                        question_base64 = selected_row['questionImage']
                        solution_base64 = selected_row['ans']
                    else:
                        question_base64 = selected_row['image']
                        solution_base64 = selected_row['ans']
                else:
                    question_base64 = selected_row['image']
                    solution_base64 = selected_row['ans']

  # Convert any non-serializable types to serializable ones
                questions.append(str(question_base64))
                solutions.append(str(solution_base64))
            except Exception as e:
                print(f"Error processing DataFrame: {e}")

        if not questions:
            print("No questions were retrieved.")
            return None, None
        
        return questions, solutions
    except Exception as e:
        print(f"Error fetching questions: {e}")
        return None, None


@app.route('/get-questions', methods=['GET'])
def get_questions():
    try:
        # Get the exam type from the request parameters (either 'jee' or 'neet')
        exam_type = request.args.get('examType', '').lower()

        if exam_type not in ['jee', 'neet']:
            return jsonify({"error": "Invalid exam type. Please specify either 'jee' or 'neet'."}), 400

        # Get random questions and solutions
        questions, solutions = get_random_questions(exam_type)

        if questions is None or solutions is None:
            return jsonify({"error": "Could not retrieve questions."}), 500

        # Return the list of base64 questions and solutions as a JSON response
        return jsonify({"questions": questions, "solutions": solutions, "status": "success"}), 200

    except Exception as e:
        print(f"Error in /get-questions route: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500
@app.route('/Jee_Maths', methods=['POST'])
def methsData():
    try:
        
        data = request.get_json()
        paragraph = data.get('paragraph', '')
        formOfDocument = int(data.get('formOfDocument', 0))
        quetionsFile = 'csvFiles/Final_Maths_Jee.csv'
        TotalMathsMergedData = 'csvFiles/TotalMathsMergedData.csv'

        result = MathsData(paragraph=paragraph, TotalMathsMergedData=TotalMathsMergedData, quetionsFile=quetionsFile, formOfDocument=formOfDocument)

        return jsonify({"message": "Data processed successfully!", "result": result}), 200

    except Exception as e:

        print("Error in processing:", e)
        return jsonify({"error": "Failed to process data"}), 500
@app.route('/Jee_Chemistry',methods=['POST'])
def chemData():
    file = r"csvFiles\Jc.csv"
    posted_data = request.get_json()

    paragraph = posted_data['text']
    
    
    return chem_data(file , paragraph)

@app.route('/Jee_Physics',methods=['POST'])
def phyData():
    file = r"csvFiles\Jp.csv"
    posted_data = request.get_json()

    paragraph = posted_data['text']
    
    return phy_data(file,paragraph)

@app.route('/NEET_bio',methods=['POST'])
def bioData():


    file = r"csvFiles\Nb.csv"

    
    posted_data = request.get_json()

    paragraph = posted_data['text']
    
    return bio_data(file,paragraph)

@app.route('/NEET_chem',methods=['POST'])
def chemNEETData():


    
    file = r"csvFiles\Nc.csv"

    
    posted_data = request.get_json()

    paragraph = posted_data['text']
    
    
    return chem_NEET_data(file,paragraph)

@app.route('/NEET_phy',methods=['POST'])
def phyNEETData():

   
    file = r"csvFiles\Np.csv"

    posted_data = request.get_json()

    paragraph = posted_data['text']
        
    
    
    return phy_NEET_data(file,paragraph)

@app.route('/GetHighLight', methods=['POST'])
def HighLight():
    if 'file' not in request.files or 'pageNumber' not in request.form:
        return jsonify({'error': 'File or page number missing'}), 400

    file = request.files['file']
    page_number = int(request.form['pageNumber'])
    return highLight(file=file , page_number=page_number)

@app.route('/Text_extract', methods=['POST'])
def extracted_text():
    file = request.get_json()
    subject = file.get("Subject")
    base_string = file.get('ImageBase64String')
    # print(base_string)
    # Add padding to the Base64 string if needed
    base_string += '=' * (-len(base_string) % 4)
    # print(base_string)
    try:
        
        result = main(base_string)
        if not isinstance(result, str):
            result = str(result)
        # print(type(result))
        if subject=="Jc":
            filePath = r"csvFiles\Jc.csv"
            return chem_data(filePath,result)
        elif subject=="Jp":
            filePath = r"csvFiles\Jp.csv"
            return phy_data(filePath,result)
        elif subject=="Nb":
            filePath = r"csvFiles\Nb.csv"
            return bio_data(filePath,result)
        elif subject=="Nc":
            filePath = r"csvFiles\Nc.csv"
            return chem_NEET_data(filePath,result)
        elif subject=="Np":
            filePath = r"csvFiles\Np.csv"
            return phy_NEET_data(filePath,result)
        elif subject=="Jm":
            filePath = r"csvFiles\Jm.csv"
            TotalMathsMergedData  =r"csvFiles\TotalMathsMergedData.csv"
            return MathsData(quetionsFile=filePath,paragraph=result , TotalMathsMergedData=TotalMathsMergedData, formOfDocument='1sd')
        else:
            return jsonify({"Sorry yarr kuch nhi h!!"}, 404) 
       
    
    except Exception as e:
        print("Error in text extraction:", e)
        return jsonify({"error": "Failed to extract text"}), 500
    
@app.route('/Jee-test', methods=['POST'])
def get_30_random_questions():
    try:
        data = request.json
        subject = data['Subject']
        test_df = pd.read_csv(rf'HacktheMountain\csvFiles\FinalTest{subject}.csv')

        # Check if there are enough questions
        if len(test_df) < 30:  # Reduce to 10 for smaller size
            return jsonify({"error": "Not enough questions available"}), 400
        
        # Randomly select 10 unique questions
        random_questions = test_df.sample(n=30).to_dict(orient='records')

        # Limit the response to image fields only
        simplified_questions = [
            {
                "image": q["image"],
                "ans": q["ans"]
            }
            for q in random_questions
        ]

        # Return the data as a JSON response
        return jsonify({
            "questions": simplified_questions,
            "total_questions": len(simplified_questions),
        }), 200

    except Exception as e:
        print(f"Error in /random-questions route: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/Neet-test', methods=['POST'])
def get_random_questions():
    try:
        data = request.json
        subject = data['Subject']
        page = int(data.get('page', 1))  # Default to page 1
        questions_per_page = 25  # Number of questions per page
        
        test_df = pd.read_csv(rf'HacktheMountain\csvFiles\FinalTest{subject}.csv')

        if len(test_df) < questions_per_page * page:
            return jsonify({"error": "Not enough questions available"}), 400

        # Randomly shuffle the dataframe
        shuffled_df = test_df.sample(frac=1).reset_index(drop=True)

        # Get the specific page of questions
        start = (page - 1) * questions_per_page
        end = start + questions_per_page
        random_questions = shuffled_df.iloc[start:end].to_dict(orient='records')

        simplified_questions = [
            {
                "image": q["image"],  # Consider sending image URLs instead of base64-encoded images to save space
                "ans": q["ans"]
            }
            for q in random_questions
        ]

        return jsonify({
            "questions": simplified_questions,
            "total_questions": len(simplified_questions),
        }), 200

    except Exception as e:
        print(f"Error in /random-questions route: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)
