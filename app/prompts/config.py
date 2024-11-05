from typing import List, Dict, Any
from . import templates

class PromptConfig:
    def __init__(
        self,
        system_content: str,
        completion_markers: List[str],
        chunking_strategy: Dict[str, Any],
        max_tokens: int = 4000,
        temperature: float = 0.7
    ):
        self.system_content = system_content
        self.completion_markers = completion_markers
        self.chunking_strategy = chunking_strategy
        self.max_tokens = max_tokens
        self.temperature = temperature

# Prompt configurations
PROMPTS = {
    "policy_json_conversion": PromptConfig(
    system_content= templates.policyMappingPrompt,
    completion_markers=["JSON_COMPLETE", "CHUNK_END"],
    chunking_strategy={
        "max_criteria_per_chunk": 100,
        "preserve_logic_boundaries": True,
        "include_context": True,
        "section_markers": [
            "PolicyInfo",
            "DrugInformation",
            "CriteriaAndLogic",
            "DosageAndAdministration",
            "Coverage",
            "ClinicalEvidence",
            "Monitoring",
            "Contraindications",
            "QuantityLimits",
            "ProviderRequirements",
            "PatientEducation"
        ]
    }
),
    "encoding": PromptConfig(
        system_content=templates.encodingPrompt,
        completion_markers=["JSON_COMPLETE", "CHUNK_END"],
        chunking_strategy={
            "max_criteria_per_chunk": 50,
            "preserve_logic_boundaries": True,
            "include_context": True
        }
    ),
    "criteria_matching": PromptConfig(
        system_content=templates.criteriaMatchingReport,
        completion_markers=["REPORT_COMPLETE"],
        chunking_strategy={
            "split_by_sections": True,
            "max_section_size": 2000
        }
    ),
    "recommendation_mapping": PromptConfig(
        system_content=templates.recommendationAndMapping + "\n" + templates.recommendationAndMappingUser,
        completion_markers=["MAPPING_COMPLETE"],
        chunking_strategy={
            "split_by_criteria": True,
            "max_criteria_per_chunk": 30
        }
    ),
    "alternative_care": PromptConfig(
        system_content=templates.alternativeCarePathway,
        completion_markers=["ALTERNATIVES_COMPLETE"],
        chunking_strategy={
            "max_alternatives_per_chunk": 3
        }
    ),
    "claim_review": PromptConfig(
        system_content=templates.claimReviewReport,
        completion_markers=["REVIEW_COMPLETE"],
        chunking_strategy={
            "split_by_sections": True,
            "max_section_size": 1500
        }
    ),
    "policy_updates": PromptConfig(
        system_content=templates.updatePolicyChangesPrompt,
        completion_markers=["UPDATES_COMPLETE"],
        chunking_strategy={
            "split_by_changes": True,
            "max_changes_per_chunk": 10
        }
    )
}

def get_prompt_config(prompt_type: str) -> PromptConfig:
    """Get configuration for a specific prompt type."""
    if prompt_type not in PROMPTS:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return PROMPTS[prompt_type]