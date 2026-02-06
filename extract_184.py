import fitz
import cv2
import numpy as np
from PIL import Image
import io


def check_checkbox(
    target_page, question_text, checkbox_x_start, checkbox_x_end
):
    """Check if a checkbox is marked by analyzing pixel density."""
    instances = target_page.search_for(question_text)
    if not instances:
        return None
    
    question_y = instances[0].y0
    
    # Extract checkbox area
    checkbox_rect = fitz.Rect(
        checkbox_x_start, question_y - 1,
        checkbox_x_end, question_y + 10
    )
    
    # Render at high resolution
    mat = fitz.Matrix(10.0, 10.0)
    pix = target_page.get_pixmap(matrix=mat, clip=checkbox_rect)
    
    # Convert to image
    img_data = pix.tobytes("png")
    img_pil = Image.open(io.BytesIO(img_data))
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to detect dark pixels (checkmarks)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Calculate fill ratio
    dark_pixels = np.sum(binary > 0)
    total_pixels = binary.size
    fill_ratio = dark_pixels / total_pixels
    
    # Threshold at 12% - below is unchecked, above is checked
    return fill_ratio > 0.12


def extract_referral_requirements(pdf_path):
    """Extract Referral Requirements with checkbox detection."""
    doc = fitz.open(pdf_path)
    
    # Find page with REFERRAL REQUIREMENTS
    target_page = None
    for page in doc:
        if "REFERRAL REQUIREMENTS" in page.get_text():
            target_page = page
            break
    
    if not target_page:
        print("Could not find REFERRAL REQUIREMENTS section!")
        doc.close()
        return
    
    # Questions and their search text
    questions = [
        (
            "Has new information been submitted since the original "
            "decision?",
            "Has new information been submitted"
        ),
        (
            "If yes, has this information been acknowledged?",
            "If yes, has this information been acknowledged"
        ),
        (
            "Have the reasons the appellant feels the decision is "
            "incorrect been addressed?",
            "Have the reasons the appellant feels"
        ),
        (
            "Are all medical investigations and assessments related "
            "to the appeal complete?",
            "Are all medical investigations"
        ),
        (
            "Is the G040 within one year of the decision?",
            "Is the G040 within one year"
        ),
        (
            "Has the G040 been converted, if applicable (e.g., LWKR, "
            "LREP, AO submission)?",
            "Has the G040 been converted"
        )
    ]
    
    # Checkbox positions (in PDF points)
    yes_checkbox_x_start = 465
    yes_checkbox_x_end = 480
    na_checkbox_x_start = 507
    na_checkbox_x_end = 522
    
    print("\n" + "="*80)
    print("REFERRAL REQUIREMENTS SECTION")
    print("="*80 + "\n")
    
    for full_question, search_text in questions:
        print(f"Q: {full_question}")
        
        # Check YES checkbox
        yes_checked = check_checkbox(
            target_page, search_text,
            yes_checkbox_x_start, yes_checkbox_x_end
        )
        
        # Check N/A checkbox
        na_checked = check_checkbox(
            target_page, search_text,
            na_checkbox_x_start, na_checkbox_x_end
        )
        
        if yes_checked is None or na_checked is None:
            print("A: Unable to detect\n")
            continue
        
        # Determine answer
        if yes_checked and not na_checked:
            answer = "YES"
        elif na_checked and not yes_checked:
            answer = "NA"
        elif yes_checked and na_checked:
            # Both checked - shouldn't happen, but pick YES
            answer = "YES"
        else:
            answer = "NO"
        
        print(f"A: {answer}\n")
    
    doc.close()
    
    print("="*80)
    print("Extraction complete!")
    print("="*80)


if __name__ == "__main__":
    extract_referral_requirements(
        "7565184_CD0122008352_1_20260126_200707.pdf"
    )
