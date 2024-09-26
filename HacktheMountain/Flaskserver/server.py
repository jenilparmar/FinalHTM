from flask import Flask, request, jsonify
from chemNEETData import chem_NEET_data
from phyNEETData import phy_NEET_data
from bioNEET import bio_data
from phyData import phy_data
from chemData import chem_data
from methsData import MathsData
from flask_cors import CORS
from highLight import highLight
from basetoimage import main
import base64 
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
