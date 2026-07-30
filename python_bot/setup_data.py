import json
import re

log_path = "/Users/omidshojaeianzanjani/.gemini/antigravity/brain/bd3e4b50-f12e-43e4-b4fb-86519d252d4f/.system_generated/logs/transcript_full.jsonl"
output_path = "/Users/omidshojaeianzanjani/Documents/Adisurc_project/document.txt"

found = False
with open(log_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f):
        try:
            data = json.loads(line)
        except:
            continue
            
        if data.get('type') == 'USER_INPUT':
            content = data.get('content', '')
            if '==Start of PDF==' in content:
                print(f"Found in line {line_num}")
                pdf_text = content.split('==Start of PDF==')[1].split('==End of PDF==')[0]
                
                # Clean up the OCR tags
                cleaned_text = re.sub(r'==Screenshot for page \d+==\n*', '', pdf_text)
                cleaned_text = re.sub(r'==Start of OCR for page \d+==\n*', '', cleaned_text)
                cleaned_text = re.sub(r'==End of OCR for page \d+==\n*', '', cleaned_text)
                
                with open(output_path, 'w', encoding='utf-8') as out:
                    out.write(cleaned_text.strip())
                print(f"Successfully extracted {len(cleaned_text)} characters to {output_path}")
                found = True
                break

if not found:
    print("Could not find the PDF text in the transcript.")
