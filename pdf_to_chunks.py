
import PyPDF2
# ok the 1st chunk of the article will be paste into the next text box

# I give information in multiple chunks. Do not  summarize the information until I ask you directly, just acknowledget that it is the nth chunk.
# Please not that the chunks may be incomplete and may continue in the next chunk.


pre_text = '. chunk: '


def pdf_to_text(pdf_file):
    with open(pdf_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text


def split_text(text, max_tokens=4000):
    chunks = []
    start = 0
    end = max_tokens

    nth = 0

    while start < len(text):
        if end < len(text) and text[end] != ' ':
            end = end - 1
        else:
            nth = nth + 1
            text_to_append = str(nth) + pre_text + '\n' + text[start:end].strip()
            chunks.append(text_to_append)
            start = end
            end = end + max_tokens

    return chunks


def save_chunks(chunks, output_prefix):
    for i, chunk in enumerate(chunks):
        with open(f'{output_prefix}_{i}.txt', 'w', encoding='utf-8') as file:
            file.write(chunk)




input_pdf = "input/rebalance_premium.pdf"
output_prefix = "output/rebalance_premium"

text = pdf_to_text(input_pdf)
chunks = split_text(text)
save_chunks(chunks, output_prefix)