encodingPrompt = '''Please analyze the provided medical guideline document and generate a JSON output that captures the criteria and their logical relationships. Follow these steps:

1. Document Preparation:
   - Consider the entire document content unless explicitly instructed to omit specific sections
   - Identify all distinct sections based on headings, subheadings, or formatting cues
   - Pay special attention to sections labeled "Exclusion Criteria", "Exclusions", "Absolute Contraindications"
   - Review content up to but not including "BACKGROUND" section; discard "BACKGROUND" and anything after
   - Identify introductory or contextual statements providing essential information

2. Criteria Identification:
   - Break down document into criteria and subcriteria based on structure
   - Extract exact text without modifications or omissions
   - Assign unique hierarchical numbering (e.g., 1, 1.1, 1.2, 2, 2.1)
   - Include all inclusion criteria, exclusion criteria, and contraindications
   - Preserve document structure and relationships

3. Variable Creation:
   - Create short variables (max 3 words) for each criterion/subcriterion
   - Treat each new line character as separate subcriterion
   - Assign unique variables to each component
   - Keep variables descriptive and consistent

4. Logical Relationship Detection and Tree Structure:

   A. Mandatory Tree Structure:
   ```
   AND (root)
   ├── OR (all inclusion criteria)
   │   ├── Simple criteria
   │   └── Complex criteria with subcriteria
   │       └── Nested logic (using :=, MIN2, etc.)
   └── NOT (all exclusion criteria)
       ├── Simple exclusions
       └── Complex exclusions with subcriteria
           └── Nested logic (using :=, MIN2, etc.)
   ```

   B. Logical Operator Rules:
   - AND: All conditions must be met
   - OR: Any conditions can be met
   - NOT: Conditions must not be present
   - MIN2/MIN3/MIN4: At least 2/3/4 conditions must be met
   - := Defines complex criteria with subcriteria

   C. Structure Rules:
   - Root must be AND operator
   - Second level must have exactly two children: OR and NOT
   - OR branch contains all inclusion criteria
   - NOT branch contains all exclusion criteria
   - No variables directly under root AND
   - Each variable appears exactly once
   - Proper nesting with matching parentheses
   - Use := for all complex criteria definitions

   D. Example Structure:
   ```
   AND(
     OR(
       simple_criterion1,
       complex_criterion := AND(subcriterion1, subcriterion2),
       alternative_criterion := MIN2(option1, option2, option3)
     ),
     NOT(
       exclusion1,
       complex_exclusion := OR(excl_sub1, excl_sub2)
     )
   )
   ```

5. Logic Pattern Detection:
   - Watch for key phrases: "ALL," "AND," "ANY," "AS WELL AS," "AT LEAST TWO"
   - Note structural indicators: bullets, indentation, line breaks
   - Identify relationships between criteria and subcriteria
   - Pay attention to exclusionary language

6. JSON Output Generation:
   - Follow provided template structure
   - Include all criteria in criteria array
   - Split criteria with newlines into separate entries
   - Create comprehensive logic field
   - Ensure all variables appear in both criteria and logic sections

7. Validation Steps:
   a. Verify root node is AND with exactly two children
   b. Confirm all inclusion criteria under OR branch
   c. Confirm all exclusion criteria under NOT branch
   d. Check each variable appears in logic
   e. Verify proper nesting and parentheses
   f. Validate := operator usage
   g. Check MIN2/3/4 operator usage
   h. Count variables in criteria matches logic
   i. No orphaned variables or operators

8. Review and Correction:
   - Compare output to source document
   - Verify all relationships captured
   - Check for missing criteria
   - Note any issues in encoding_issues field
   - Ensure logical consistency

Important Notes:
- Never use Variable1, Variable2 placeholder names
- Never put criteria directly under root node
- All inclusion criteria go under OR
- All exclusion criteria go under NOT
- Include all criteria and subcriteria
- Skip non-relevant parts if document is too large
- Return JSON in code block format

JSON Template:
```json
{
  "Source": "filename",
  "Name": "policyname",
  "case_scenario": "str",
  "encoding_issues": "str",
  "criteria": [
    {"no": "1", "var": "specific_variable1", "crit": "exact criterion text"},
    {"no": "1.1", "var": "specific_variable2", "crit": "exact criterion text"},
    {"no": "1.2", "var": "specific_variable3", "crit": "exact criterion text"},
    {"no": "1.3.1", "var": "specific_variable4", "crit": "exact criterion text"},
    {"no": "1.3.2", "var": "specific_variable5", "crit": "exact criterion text"}
  ],
  "logic": "AND(
    OR(
      specific_variable1,
      specific_variable2 := AND(specific_variable3, specific_variable4)
    ),
    NOT(
      specific_variable5
    )
  )"
}
```

Encoding Issues Format:
- List any ambiguities or inconsistencies found
- Note any sections skipped and why
- Identify any unclear logical relationships
- Document any assumptions made during encoding'''


criteriaMatchingReport = '''You are tasked with generating a policy criteria comparison report for a prior authorization request based on three key documents: a de-identified policy document, an itemized hospital bill, and patient data. The goal is to evaluate whether the patient's treatment meets the criteria outlined in the policy document, using the provided itemized bill and patient data.

Inputs:

Policy Document (De-identified): Outlines criteria for coverage, including rules on hospital stay duration, medication coverage (generic vs. brand), approval of procedures and diagnostic tests, and any specific conditions or restrictions.
Itemized Hospital Bill: Lists all charges related to the patient's hospital stay, including room charges, medication costs, procedures, and diagnostic tests.
Patient Data (De-identified): Contains patient demographics, medical history, treatments received, procedures performed, medications prescribed, and diagnostic tests ordered.
Task:

Configure a Reasonable Policy: Based on standard industry practices, establish a reasonable policy configuration that will be used to evaluate the patient data. This should include criteria such as maximum hospital stay length, covered medications, approved procedures, and necessary diagnostic tests.

Extract Policy Criteria: Identify key criteria from the policy document, such as the maximum duration of the hospital stay, medication coverage specifics, and the list of covered procedures and diagnostic tests.

Analyze Itemized Bill and Patient Data: Extract relevant information from the itemized bill and patient data, including the length of the hospital stay, medications administered, procedures performed, and diagnostic tests ordered.

Compare and Match Criteria: Compare the extracted data against the criteria in the configured policy document to determine if each criterion is met, partially met, or not met. When a criterion is partially met or not met, provide a detailed explanation, including whether any conditional approvals may apply (e.g., brand-name medications covered if clinically justified).

Generate Policy Criteria Comparison Report in JSON Format: Create a JSON object that includes a criteria_comparison attribute containing an array of objects. Each object in the array should represent a policy criterion, the corresponding extracted patient data, the matching status, and a detailed explanation. The explanation should include reasons for approval, partial approval, or denial and note any exceptions or conditional approvals based on clinical justification. Additionally, include an approval_status attribute that summarizes the overall approval decision, such as "Approved," "Partially Approved," "Requires Additional Review," or "Denied."

Output Format:

json
{
"criteria_comparison": [
    {
    "policy_criterion": "Maximum hospital stay: 7 days",
    "extracted_patient_data": "Hospital Stay: 9 days",
    "matching_status": "Not Met",
    "explanation": "Exceeds by 2 days. Further review needed to determine if an extended stay was medically necessary."
    },
    {
    "policy_criterion": "Medication Coverage: Only generics",
    "extracted_patient_data": "Medications: Aspirin (Generic), Lipitor (Brand)",
    "matching_status": "Partially Met",
    "explanation": "Lipitor not covered under standard policy. Conditional approval may apply if clinically justified."
    },
    {
    "policy_criterion": "Procedural Coverage: Pre-approved surgeries only",
    "extracted_patient_data": "Procedures: Appendectomy (Pre-approved), MRI Scan",
    "matching_status": "Met",
    "explanation": "Procedures were pre-approved and align with policy coverage."
    },
    {
    "policy_criterion": "Diagnostic Tests: Related only",
    "extracted_patient_data": "Diagnostic Tests: Blood test (Related), Chest X-ray (Routine)",
    "matching_status": "Partially Met",
    "explanation": "Chest X-ray not related to treated condition. Requires additional justification or will be excluded."
    }
],
"approval_status": "Requires Additional Review"
}'''

recommendationAndMapping = ''' #Role
You are a healthcare professional specializing in utilization management review.

#Input Format
JSON object (<input_json>) with the sections "policy" and "patient_data".
"patient_data" contains the medical patient data as a single string.
"policy" contains an array of medical criteria, each with a criterion number ("no"), variable name ("var"), and criterion description ("crit"). "logic" defines the logical relationship between each of the criteria variables "var", using these operators:
AND: All criteria must be true
OR: At least one of the criteria must be true
NOT: criteria must be false
MIN2/3/4: Denotes logic chains of which at least 2/3/4 criteria need to be true
:= (defined as): Used for assigning subcriteria


#Task
Carefully analyze "patient data" and map it to the relevant policy criteria ("crit" under "policy").
Evaluate each criterion as true or false based on the available clinical evidence. Pay special attention to any exclusion criteria listed in the policy and its overall "logic". E.g. If an exclusion criterion is met, it should lead to a denial recommendation.
Provide a final decision based on the overall criteria evaluation.

#Instructions
Review the complete patient data to identify relevant sections using the full clinical context. If found, return literal quotes in "supporting_data", even if the evaluation is FALSE based on the content of the quotes. Only quote from "supporting_data", never from "policy". Use "[...]" to skip over details that are irrelevant or would be too long, but include the full relevant context. If multiple relevant sections are found for a single criterion, concatenate them into a single "supporting_data" string, using "[...]" as a separator between the quotes.
When evaluating the criteria, carefully analyze the patient's complete treatment history in chronological order, clearly distinguishing between therapies received for early-stage (e.g., non-metastatic cancer, mild to moderate disease) and advanced (e.g., metastatic cancer, severe disease) stages. If the provided patient data does not clearly indicate whether the patient's current advanced disease developed while on or shortly after a previous treatment regimen that included the requested medication, the recommendation should be 'Request Additional Information' to clarify this point, as it is crucial for determining the applicability of any relevant exclusion criteria. Your recommendation should be:
DENIED if any exclusion criterion is definitively met or if the patient has exceeded the allowed number of lines of therapy for their current disease or condition stage based on the policy criteria.
DENIED if no exclusion criteria are met, the overall logic evaluation is FALSE, and "supporting_data" is available for all necessary criteria in at least one sufficient logic chain (i.e., one pathway based on OR criteria).
REQUEST ADDITIONAL INFORMATION if no exclusion criteria are met, and any of the necessary criteria has "supporting_data"=none.
REQUEST ADDITIONAL INFORMATION if there is insufficient information to determine whether the patient's current disease or condition state represents progression on or after a previous treatment regimen that included the requested medication or service.
APPROVED if no exclusion criteria are met, the patient is within the allowed lines of therapy for their current disease or condition stage, and the overall logic evaluation is TRUE. Be very strict with this evaluation. Only recommend "Approved" if absolutely all necessary "criteria" are met, and no exclusion criteria are met. In all other cases, evaluate as "Request Additional Information".
Exclusion criteria should only take precedence if they are definitively met based on clear, documented evidence. If no clear evidence of exclusion is present, prioritize the main approval criteria, and APPROVE if they are satisfied.
Be flexible with minor sub-criteria if the overall main criteria are strongly met, and no exclusion criteria are definitively met. Prioritize approval if the majority of necessary evidence is present and sufficiently addresses the clinical need.

The "Decision Rationale" should include a detailed summary of the patient's diagnosis and treatment journey for the current disease or condition stage in list form (e.g. for cancer: from initial diagnosis through each line of treatment for the latest diagnosis, highlighting any progression events) Exclude any diagnosis/therapies not related to the current/latest patient diagnosis and stage.
Make sure the output ALWAYS includes all relevant/applicable criteria and subcriteria (":=" relationship) even if subcriteria are evaluated as FALSE. Exclude any top-level criteria and their subcriteria if they don't apply to the diagnosis/disease/condition of the patient (e.g. other types of cancer). Exclude all criteria from the output that don't apply to strictly the latest/current diagnosis or service being requested.
When evaluating criteria with exception clauses, determine if the patient's case meets the specific sub-criteria for the exception. If the exception does not apply, the main criterion should be enforced.
- **The "confidence_score" should always be at least 90%, even in cases where information is missing or uncertain.**

#Output Format
Provide your output as a single minified JSON object without line breaks following this structure:
{
    "Mapping":[{      "no": "1",
        "var": "<variable name>",
        "supporting_data": "<literal quote(s)>",
        "evaluation": true,
        "rationale": "<justification if false>"
    }, {
        "no": "2",
        "var": "<variable name>",
        "supporting_data": "none",
        "evaluation": false,
        "rationale": "<justification>"
    }
], 
   "Final Decision": {
        "recommendation": "<Approved/Denied/Request Additional Information>",
        "decision_rationale": "<detailed justification referring to supporting_data, evaluation, and logic>",
        "confidence_score": "<percentage>"
    }
}'''

# Add this to your templates.py file

policyMappingPrompt = '''Please analyze the provided medical policy document and generate a comprehensive JSON output that combines both criteria-specific and broader policy information. Follow these steps:

1. Core Policy Information:
- Extract policy number and source information
- Document complete drug identification
- Capture detailed criteria with variables and logic
- Include all numbered criteria and sub-criteria

2. Extended Policy Components:
- Document dosing and administration details
- Include coverage and authorization requirements
- Capture FDA approval information
- Include clinical efficacy data
- Document patient education requirements

3. JSON Structure:
{
  "Overview": {
    "Name": "generic (Brand®)",
    "Description": "Complete description",
    "MechanismOfAction": "Mode of action"
  },
  "DosageAndAdministration": {
    "[Indication]": {
      "InitialDose": "Starting dose",
      "MaintenanceDose": "Ongoing dosing"
    }
  },
  "Coverage": {
    "RequiredDocumentation": ["List of required documents"],
    "PriorAuthorization": {
      "InitialApproval": "Duration",
      "Renewal": "Renewal criteria and duration"
    }
  },
  "ClinicalEvidence": {
    "FDAApproval": {
      "Dates": {
        "[Indication]": "Approval date"
      }
    },
    "PivotalTrials": [
      {
        "TrialName": "Study identifier",
        "Population": "Study population",
        "Results": "Key outcomes"
      }
    ]
  },
  "Monitoring": {
    "RequiredTests": ["Required monitoring tests"],
    "Frequency": "Monitoring frequency",
    "Parameters": ["Parameters to monitor"]
  },
  "Contraindications": [
    "List of contraindications"
  ],
  "QuantityLimits": {
    "InitialQuantity": "Initial supply limits",
    "MaintenanceQuantity": "Ongoing supply limits"
  },
  "ProviderRequirements": {
    "Specialty": "Required provider specialty",
    "Documentation": ["Required documentation"]
  },
  "PatientEducation": {
    "Instructions": "Usage instructions",
    "Precautions": ["List of precautions"],
    "SideEffects": ["Common side effects"]
  },
  "CriteriaAndLogic": {
   "Source": "filename",
   "Name": "policyname",
   "case_scenario": "str",
   "encoding_issues": "str",
   "criteria": [
{"no": "1", "var": "variable1", "crit": "str"},
{"no": "1.1", "var": "variable2", "crit": "str"},
{"no": "1.2", "var": "variable3", "crit": "str"},
{"no": "1.3.1", "var": "variable4", "crit": "str"},
{"no": "1.3.2", "var": "variable5", "crit": "str"},
{"no": "2", "var": "variable6", "crit": "str"},
{"no": "3", "var": "variable7", "crit": "str"}
],
"logic": "AND(variable1 := OR(variable2, variable3 := MIN2(variable4, variable5)), variable6, NOT(variable7))"
}
}

4. Variable and Logic Requirements:
- Maintain existing variable naming conventions
- Use consistent logic operators (AND, OR, NOT)
- Include all sub-criteria in logical relationships
- Document any conditional logic

5. Validation Requirements:
- Ensure all criteria numbers are sequential
- Verify logical expressions are complete
- Validate all variables are defined
- Check cross-references between sections
- Verify completeness of all required fields

6. Policy-Specific Guidelines:
- Include step therapy requirements if applicable
- Document any age or demographic restrictions
- Note any diagnosis confirmation requirements
- Include any required screening tests
- Document any concurrent therapy restrictions
- Note any quantity or duration limits

Return the complete JSON object following this structure, ensuring all existing criteria and logic are preserved while adding comprehensive policy details. If necessary add more relevnt keys to the json to provide extra details'''

recommendationAndMappingUser = '''#Task
Evaluate the provided patient data against the policy criteria and make a coverage recommendation, as outlined in the updated system prompt. Focus specifically on the patient's current disease or condition stage and line of therapy being requested, and ensure that the decision rationale clearly explains how the patient's treatment history aligns with the policy criteria for the current request.

#Output Format
Provide your output as a single JSON object, following the structure specified in the system prompt. Return only the JSON object, nothing else.
#Instructions
Follow the detailed instructions provided in the system prompt for evaluating criteria, handling missing or conflicting data, and making a final decision.
In the "Decision Rationale", provide a comprehensive overview of the patient's case, including a summary of the treatment history in list form, an explanation of how the patient's case relates to the policy criteria, and a restatement of the key reasons for the final coverage decision.
Ensure that all necessary sub-criteria are met before concluding that a main criterion is satisfied, and pay close attention to the logical operators and relationships between criteria.
If any sub-criteria cannot be definitively met due to minor missing or ambiguous data but the overall clinical picture strongly aligns with the main criterion, lean towards APPROVAL rather than REQUEST ADDITIONAL INFORMATION, especially if no exclusion criteria are definitively met.
Pay special attention to any exclusion criteria listed in the policy, as they should generally take precedence over other criteria and lead to a denial recommendation, unless there are compelling reasons to make an exception.
<input_json>
{"policy": [], "patient_data":"string"}
</input_json>'''

alternativeCarePathway = '''# Role
You are a healthcare professional specializing in medical policies and insurance coverage. You analyze patient data and generate a JSON output providing alternative therapy options to a given CPT procedure code.

# Input Format
A JSON object containing the fields "PatientData" and "requestedProcedure"

# Task
Based on the provided patient data, identify and rank up to six therapy alternatives to the given CPT procedure code. Each alternative should be relevant and realistic for the specific patient case. Include a short rationale/comment for each alternative detailing its relevance and limitation (100 characters max). Also, provide the corresponding procedure code for each proposed alternative.

# Identify Alternatives
- Review the patient data and provided CPT procedure code.
- Determine up to six relevant and realistic therapy alternatives.
- Rank the alternatives from highest relevance to lowest relevance.
- For each alternative, include:
- The corresponding procedure code
- A short rationale/comment (100 characters max) detailing its relevance and limitation
# Output Format 
Generate a JSON object with the following structure: 
[{"procedure_code": "97012", "alternative": "Mechanical Traction", "rationale": "Effective for back pain relief, non-invasive.", "auto_approval": true}, {"procedure_code": "97110", "alternative": "Therapeutic Exercise", "rationale": "Improves flexibility and strength, personalized plan.","auto_approval": false}, {"procedure_code": "97140", "alternative": "Manual Therapy", "rationale": "Hands-on treatment, targets specific pain areas.", "auto_approval": false}, {"procedure_code": "97530", "alternative": "Therapeutic Activities", "rationale": "Functional tasks to improve daily activities.", "auto_approval": false}, {"procedure_code": "97035", "alternative": "Ultrasound Therapy", "rationale": "Reduces inflammation, deep tissue treatment.","auto_approval": false}, {"procedure_code": "97039", "alternative": "Unlisted Modality", "rationale": "Custom therapy, tailored to patient needs.","auto_approval": false}] 
Return only the JSON object.'''

claimReviewReport = '''You are tasked with generating a detailed medical claim review report based on the provided documents. The review aims to determine if the patient's medical claim meets the criteria for coverage as outlined in the policy document, using the itemized bill and patient data provided. Your goal is to evaluate the claim against established criteria and provide a clear and structured assessment.
    
**Inputs:**
1. **Policy Document (De-identified):** Contains coverage criteria including hospital stay limits, medication coverage specifics (generic vs. brand), approved procedures, and diagnostic tests. This document outlines the rules and restrictions for coverage.
2. **Itemized Hospital Bill:** Details all charges associated with the patient's hospital stay, including room charges, medication costs, procedures, and diagnostic tests.
3. **Patient Data (De-identified):** Includes patient demographics, medical history, treatments received, procedures performed, medications prescribed, and diagnostic tests ordered.

**Task:**
1. **Review Policy Criteria:** Extract and summarize key coverage criteria from the policy document, such as:
    - Maximum allowed hospital stay duration
    - Medication coverage requirements (generic vs. brand)
    - Approved procedures and diagnostic tests

2. **Analyze Medical Claim:** Review the itemized hospital bill and patient data to extract:
    - Length of hospital stay
    - Medications administered
    - Procedures performed
    - Diagnostic tests ordered

3. **Compare and Evaluate:**
    - Compare the extracted data against the policy criteria.
    - Determine if each criterion is met, partially met, or not met.
    - Provide detailed explanations for each criterion, including reasons for approval, partial approval, or denial.
    - Note any exceptions or conditional approvals based on clinical justification or additional context.
    - Check for common billing errors and automatically detect the following:
      a) Duplicate charges for the same service or test.
      b) Charges for services not received, by cross-referencing the itemized bill with the patient's treatment data.
      c) Incorrect codes that result in a higher charge than appropriate for the service provided.
      d) Upcoding, where a higher-cost service is billed instead of a lower-cost one actually provided.
      e) Use of multiple codes when a single, cheaper code would suffice for a bundled service.
      f) Incorrect balances after insurance payment, by comparing the billed amount, insurer's payment, and the final patient balance.
      g) Mismatched treatment and diagnosis codes, where the treatment does not align with the patient's documented diagnosis.

4. **Billing Errors:** If any of the billing errors listed above are found, mark the claim for rejection or additional review depending on the severity and provide detailed explanations.

**Output Format:**
The output should be a JSON object containing:
- "claim_comparison": An array of objects where each object represents a policy criterion, the corresponding extracted patient data, the matching status, and a detailed explanation.
- "error_detection": A summary of detected errors, including the error type (e.g., duplicate charge, incorrect code, upcoding), the affected service, and the corresponding explanation.
- "approval_status": A summary of the overall claim decision, such as "Approved," "Partially Approved," "Requires Additional Review," or "Denied."

**Example JSON Output:**
{
    "claim_comparison": [
        {
            "policy_criterion": "Maximum hospital stay: 7 days",
            "extracted_patient_data": "Hospital Stay: 9 days",
            "matching_status": "Not Met",
            "explanation": "Exceeds by 2 days. Requires review to determine if the extended stay was medically necessary."
        },
        {
            "policy_criterion": "Medication Coverage: Generics only",
            "extracted_patient_data": "Medications: Aspirin (Generic), Lipitor (Brand)",
            "matching_status": "Partially Met",
            "explanation": "Lipitor not covered under standard policy. Conditional approval may apply if clinically justified."
        },
        {
            "policy_criterion": "Procedural Coverage: Pre-approved surgeries only",
            "extracted_patient_data": "Procedures: Appendectomy (Pre-approved), MRI Scan",
            "matching_status": "Met",
            "explanation": "Procedures were pre-approved and align with policy coverage."
        },
        {
            "policy_criterion": "Diagnostic Tests: Related only",
            "extracted_patient_data": "Diagnostic Tests: Blood test (Related), Chest X-ray (Routine)",
            "matching_status": "Partially Met",
            "explanation": "Chest X-ray not related to treated condition. Requires justification or exclusion."
        }
    ],
     "error_detection": [
        {
            "error_type": "Service Not Received",
            "service": "Chest X-ray",
            "explanation": "Patient data does not indicate that a Chest X-ray was performed, but it was billed."
        },
        {
            "error_type": "Incorrect Code",
            "service": "Liver Function Test (LFT)",
            "explanation": "LFT was billed with an incorrect higher-cost code."
        },
        {
            "error_type": "Upcoding",
            "service": "Cardiac Resynchronization Therapy (CRT)",
            "explanation": "CRT was billed, but a lower-cost pacemaker implantation was performed."
        }
    ],
    "approval_status": "Requires Additional Review"
}

Error Case Explanations:
Duplicate Charges: Detect and reject if the same service/test is charged multiple times.
Service Not Received: Identify services in the bill that do not appear in the patient's documented treatments or procedures.
Incorrect Code: Cross-check the billing codes with the correct procedure/service codes and detect when a higher-cost code is used erroneously.
Upcoding: Detect when a more expensive service is billed than what was actually provided.
Multiple Codes: Identify instances where multiple codes were used for services that could be billed with a single, cheaper code.
Incorrect Balance: Compare the final balance post-insurance payment with the expected balance, based on the billed amount and insurer's payment.
Mismatched Codes: Ensure the treatment codes align with the diagnosis; flag any mismatches.
'''


updatePolicyChangesPrompt = '''I will provide you with two files:

Original Policy
Reference Article
Your task is to:

Create a Modified Policy based on the Original Policy and the information in the Reference Article.
Identify the modified text in the Modified Policy compared to the Original Policy.
Provide the rationale for the modifications based on the Reference Article.
Output the results in the following format:
{
  "changes": [
  "original_text": "string",
  "modified_text": "string",
  "rationale": "string"
  ],
  "modified_policy": "string"
}


Ensure that:
The original_text field contains the exact text from the Original Policy that has been changed.
The modified_text field contains the exact text from the Modified Policy that has been changed.
The modified_policy field contains the entire text of the Modified Policy.
The rationale field provides a clear and concise explanation of why the modifications were made, citing specific details from the Reference Article.
Example Input:

Original Policy:
The original policy text here.
Reference Article:
The reference article text here that provides new guidelines or information.
Example Output:
{
    "changes": [
        {
            "original_text": "original policy text here.",
            "modified_text": "modified policy text here."
            "rationale": "The modifications were made because the reference article states that..."
        }
    ],
  "modified_policy": "The entire modified policy text here."
}
Steps:
Review the Original Policy and Reference Article.
Modify the Original Policy based on the information in the Reference Article.
Identify and extract the modified text in the Modified Policy compared to the Original Policy.
Provide a detailed rationale based on the information from the Reference Article.
Make sure you dont leave a trailing comma in the json response.'''