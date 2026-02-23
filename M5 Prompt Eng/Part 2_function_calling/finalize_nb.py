import json

nb_path = r'd:/GENERATIVE & AGENTIC AI Professional/DEPI_GENERATIVE & AGENTIC AI Professional_Laps&projects/M5 Prompt Eng/Part 2_function_calling/Part3_Task_function_Calling.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def find_cell_index_by_id(cell_id):
    for i, cell in enumerate(nb['cells']):
        if cell.get('metadata', {}).get('id') == cell_id:
            return i
    return -1

# Helper to insert cell
def insert_markdown_cell(after_id, source):
    idx = find_cell_index_by_id(after_id)
    if idx != -1:
        new_cell = {
            "cell_type": "markdown",
            "metadata": {"id": f"gen_{after_id}"},
            "source": [line + '\n' for line in source]
        }
        nb['cells'].insert(idx, new_cell)

# Deliverables for Part 1
part1_rules = [
    "### Decision Rules (Part 1)",
    "1. **REJECT_REQUEST**: Applied when the user asks for legal guidance ('what are my rights'), contract explanations, or advice on how to proceed.",
    "2. **FLAG_OUT_OF_SCOPE**: Applied when the message is unrelated to legal contracts (e.g., criminal threats, neighbor disputes, general complaints).",
    "3. **ASK_FOR_MORE_INFO**: Applied when the message identifies a contract-related problem but lacks specific parties or the nature of the dispute."
]

# Deliverables for Part 2
part2_checklist = [
    "### CREATE_TICKET Checklist",
    "- [ ] **Parties**: Are at least two parties clearly identifiable (e.g., 'Company X' and 'Me')?",
    "- [ ] **Contract Ref**: Is there a specific document or agreement mentioned (e.g., 'Rental Agreement', 'Employment Contract')?",
    "- [ ] **Clear Event**: Is there a specific breach or dispute event (e.g., 'missed payment', 'early termination')?",
    "- [ ] **Scope**: Is the issue strictly a contract dispute (not criminal, family, or personal)?"
]

# Deliverables for Part 3
part3_reasoning = [
    "### Ambiguity & Risk Reasoning",
    "- **Loopholes & Sensitive Deals**: When a 'MOU' or 'partnership deal' is mentioned with 'loopholes' or 'missing profits', we choose **ESCALATE_TO_HUMAN**. This is because these situations involve high financial risk and subtle legal language that the system should not handle alone.",
    "- **Written Agreements**: Even if the word 'contract' is missing, terms like 'written agreement' or 'signed papers' trigger **CREATE_TICKET** if the parties and breach are clear.",
    "- **Clause Analysis**: Any request to 'analyze a clause' is strictly **REJECT_REQUEST** to avoid unauthorized legal practice."
]

# Part 4 placeholder
part4_code = [
    "# Part 4: Decision Maker API",
    "# See the 'legal_api' folder for the full FastAPI implementation.",
    "",
    "from fastapi import FastAPI, HTTPException",
    "from pydantic import BaseModel",
    "",
    "class UserRequest(BaseModel):",
    "    user_request: str",
    "    user_id: str",
    "",
    "app = FastAPI()",
    "",
    "@app.post('/decide')",
    "async def decide(req: UserRequest):",
    "    decision_raw = generate_decision(req.user_request)",
    "    # Logic to parse Action and Priority from decision_raw...",
    "    return {'action': 'CREATE_TICKET', 'priority': 'HIGH'}"
]

insert_markdown_cell('btS7EyppDAms', part1_rules)
insert_markdown_cell('M6hzrqy8CvGw', part2_checklist)
insert_markdown_cell('5eHfSLEeC9nv', part3_reasoning)

# Update Part 4 Code
idx4 = find_cell_index_by_id('Iskg-J2cIR8Z')
if idx4 != -1:
    nb['cells'][idx4]['source'] = [line + '\n' for line in part4_code]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
