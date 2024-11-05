
# Define the data models
class PolicyCriterion(BaseModel):
    no: int
    crit: str
    desc: str


class PolicyCriteria(BaseModel):
    criteria: List[PolicyCriterion]
    logic: str


class PolicyDocument(BaseModel):
    description: str
    application_drug_list: str
    policy_criteria: PolicyCriteria
    contraindications: str
    quantity_level_limit: str

