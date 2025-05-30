## Question-Answer Inverse (QAI)

QAI is a data augmentation technique that converts single-choice QA into multi-choice QA through question-answer inverse. Here's how it works:

### Basic Conversion Example
**Original Question from STAR:**  
What did the person do with the clothes?  
A. Lied on  
B. Took  
C. Put down ✅  
D. Washed  

**Inverted Question:**  
What **didn't** the person do with the clothes?  
A. Lied on ✅  
B. Took ✅  
C. Put down  
D. Washed ✅  

### Random Drop Mechanism
To prevent the model from learning false patterns (e.g., always selecting three answers for negative questions), we randomly drop correct answers:

**Possible Inverted Variant:**  
What didn't the person do with the clothes?  
A. Lied on ✅  
B. Took ✅  
C. Put down  

### Implementation
Scripts for QAI conversion are available in:
```bash
python question_answer_inverse/convert_nextgqa.py  # For NExT-GQA inversion
python question_answer_inverse/convert_star.py     # For STAR inversion
```

Output files will be generated in the `evaluation/` directory:
- NExT-GQA: `nextgqa_val_mixed.json`
- STAR: `STAR_mixed.json`