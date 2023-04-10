import os
import openai
import PyPDF2
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from pdf2image import pdfinfo_from_path
from concurrent.futures import ThreadPoolExecutor
import tiktoken 


# ISSUES: 
# - Possibly important disjoint sentences at the 
#   beginning or end of the chunks.
# 
# - Cannot read sideways pages (generates garbage)
#
# - Transcribes all garbage from any and every point on page
#  
# 
# - Does not always read word chunks in correct order if columns are strange


# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'<insert Tesseract path here>' #ex: /usr/local/bin/Tesseract


### FUNC: IMAGE TO TEXT ###
# Function to convert PDF page to image and perform OCR
def pdf_page_to_text(page_number):
    image = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)[0]
    text = pytesseract.image_to_string(image)
    return text



### FUNC: TOKEN COUNTER ###
# Counts tokens in a string of text
def count_tokens(countIt: str):
    # The first time this runs, it will require an 
    # internet connection to download. Later runs 
    # won't need an internet connection.
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(countIt)
    return len(tokens)





### OCR Stuff ###
# Provide the path to the PDF file
pdf_path = r'<insert pdf path here>' #ex: /Users/liz/folder/foo.pdf
pdf_info = pdfinfo_from_path(pdf_path)
number_of_pages = pdf_info['Pages']
# !!!!! change begin and end pages depending on pdf
beginAtPage = 2
endAtPage = 11+1 # number_of_pages+1 means append every page
# Extract text from the PDF
with ThreadPoolExecutor() as executor:
    pages_text = executor.map(pdf_page_to_text, range(beginAtPage, endAtPage))
extracted_text = " ".join(pages_text)




### CHATGPT Stuff ###
# Set up the OpenAI API client (API key needed)
openai.api_key = " <insert API key here> "
# set-up 
responses = []
chunks = []
# !!!!! change prompt as needed
prompt = "<insert prompt here>: \n"
textTok = count_tokens(extracted_text)
promptTok = count_tokens(prompt)
# *** response and ask tokens must add to be <= 8192 (nbrs below are highly experimental)
responseTokens = 6500 
askTokens = 1692
div = (textTok // (askTokens - promptTok)) + 1
# sets start and end of first text chunk
start = 0
end = len(extracted_text) // div



### RUNNING TRANSCRIBED ARTICLE THROUGH CHATGPT ###
iterCount = 0
while end <= len(extracted_text):
    # print(count_tokens(extracted_text[start : end]))
    iterCount += 1
    chunks.append(extracted_text[start : end])
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Replace with the desired GPT model
            messages=[{
                "content": prompt + chunks[len(chunks)-1],
            "role": "user"}],
            max_tokens=responseTokens ,  # Adjust this value based on the desired response length
            n=1,
            temperature=0.5,
        )
    except: # openai.error.RateLimitError:
        print("Exhausted at iteration " + str(iterCount) + " out of " + str(div)) 
        print("I fell asleep reading. I'm overworked and underpaid.")
        break

    responses.append(response)
    start = end
    end = end + (len(extracted_text) // div)




### RESULTS PRINTING/DOCUMENTATION ###
strAsk = str(askTokens)
strResp = str(responseTokens)
print(" Prompt tokens: " + strAsk + "  Response tokens: " + strResp + "\n")
with open('Result', 'a', encoding='utf-8') as output_file:
    output_file.write(" Prompt tokens: " + strAsk + "  Response tokens: " + strResp + "\n")
    output_file.write(" Prompt: " + prompt + "\n \n")
    for i in responses:
        generated_response = i.choices[0].message.content
        output_file.write(" RESPONSE: \n")
        output_file.write(generated_response)
        output_file.write("\n \n \n")
    output_file.write("\n\n\n\n\n\n\n\n\n\n")
