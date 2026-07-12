"""
Persona generator: converts cluster statistics into rich human-readable profiles
using Ollama LLM. Falls back to template-based personas when Ollama is unavailable.
"""

import json
import re
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import logger
from utils.ollama_client import ollama


PERSONA_SYSTEM_PROMPT = """You are a senior consumer insights analyst writing persona profiles.
Given cluster statistics, generate a realistic, data-driven persona.
Respond ONLY with a valid JSON object with these keys:
- name: string (first + last name, creative but professional)
- tagline: string (one punchy sentence describing this persona)
- motivations: array of 3 strings
- pain_points: array of 3 strings
- recommended_actions: array of 3 business actions
- narrative: string (2-3 sentence description of this person's story)
No markdown. No explanation. Only the JSON object."""


def _build_persona_prompt(segment_id: int, stats: dict) -> str:
    return f"""Segment {segment_id} statistics:
- Average LTV: ${stats.get('avg_ltv', 0):.0f}
- Churn Risk: {stats.get('avg_churn_risk', 0)*100:.1f}%
- Avg Total Spend: ${stats.get('avg_total_spend', 0):.0f}
- Avg Purchase Frequency (monthly): {stats.get('avg_purchase_frequency', 0):.1f}
- Avg NPS Score: {stats.get('avg_nps', 0):.1f}/10
- Avg Session Count (30d): {stats.get('avg_sessions', 0):.0f}
- Avg Age: {stats.get('avg_age', 35):.0f}
- Top Income Bracket: {stats.get('top_income', 'medium')}
- Top Subscription: {stats.get('top_subscription', 'basic')}
- Segment Label: {stats.get('label', f'Segment {segment_id}')}

Generate the persona JSON."""


FALLBACK_PERSONAS = [
    {
        "name": "Alexandra Chen",
        "tagline": "The loyal power-buyer who expects premium experience in every interaction.",
        "motivations": ["Exclusive access to premium products", "Seamless omnichannel experience", "Recognition for loyalty"],
        "pain_points": ["Price increases without added value", "Inconsistent service quality", "Slow support response times"],
        "recommended_actions": ["Enroll in VIP loyalty tier", "Offer early access to new products", "Assign dedicated account manager"],
        "narrative": "Alexandra has been a customer for over three years, consistently purchasing across electronics and home categories. Her high NPS score reflects genuine brand advocacy, making her a prime candidate for referral programs.",
    },
    {
        "name": "Marcus Williams",
        "tagline": "The deal-hunter ready to churn unless he finds compelling value.",
        "motivations": ["Best price guarantee", "Flash sale notifications", "Flexible return policy"],
        "pain_points": ["Feels undervalued compared to new customers", "Lacks personalized offers", "High shipping costs erode perceived value"],
        "recommended_actions": ["Send targeted win-back offer", "Highlight loyalty pricing", "Offer free shipping threshold reduction"],
        "narrative": "Marcus shops infrequently but spends significantly when he does. His purchase recency has been declining, signaling he's evaluating competitors. A timely, personalized discount could re-anchor his loyalty.",
    },
    {
        "name": "Priya Patel",
        "tagline": "The newly onboarded explorer discovering her preferred product mix.",
        "motivations": ["Product discovery and variety", "Peer reviews and social proof", "Easy onboarding experience"],
        "pain_points": ["Overwhelmed by product choice", "Unclear loyalty benefits", "Hasn't found trusted product categories yet"],
        "recommended_actions": ["Trigger onboarding email sequence", "Recommend best-sellers by category", "Offer first-purchase incentive"],
        "narrative": "Priya joined recently through a social media ad. She's browsing frequently but her cart abandonment rate is high. Guiding her to top-rated products in her interest areas could accelerate her first major purchase.",
    },
    {
        "name": "David Okafor",
        "tagline": "The dormant account with recoverable lifetime value.",
        "motivations": ["Trusted quality assurance", "Convenient reordering", "Budget-conscious deals"],
        "pain_points": ["Found better pricing elsewhere", "Experienced a service failure", "Lost interest over time"],
        "recommended_actions": ["Launch re-engagement campaign", "Survey on departure reason", "Offer one-time win-back discount"],
        "narrative": "David was an active customer 18 months ago before his engagement dropped significantly. A service incident in his history suggests a targeted apology with compensation could reactivate him.",
    },
    {
        "name": "Sarah Thompson",
        "tagline": "The budget-focused buyer maximizing value at every touchpoint.",
        "motivations": ["Transparent pricing with no surprises", "Useful product comparisons", "Bulk or bundle savings"],
        "pain_points": ["Hidden fees at checkout", "Too many upsell prompts", "Wish list items going out of stock"],
        "recommended_actions": ["Promote subscription savings bundles", "Enable price-drop alerts", "Offer buy-more-save-more tiers"],
        "narrative": "Sarah is methodical: she researches thoroughly, compares options, and only buys when she's confident in value. Low session-to-purchase ratio reflects deliberate decision-making rather than disengagement.",
    },
    {
        "name": "James Nguyen",
        "tagline": "The brand advocate whose word-of-mouth drives measurable acquisition.",
        "motivations": ["Being first to experience new features", "Community belonging", "Recognition for referrals"],
        "pain_points": ["Referral program lacks transparency", "Wants more co-creation opportunities", "Premium features lag expectations"],
        "recommended_actions": ["Invite to beta tester program", "Feature in community spotlight", "Increase referral commission tier"],
        "narrative": "James consistently gives 9-10 NPS scores and actively shares products on social platforms. His network effect has already driven measurable sign-ups. Formalizing his advocacy through a structured program would amplify impact.",
    },
]


def _parse_ollama_response(response: str, segment_id: int, fallback: dict) -> dict:
    """Extract JSON from Ollama response, falling back to template."""
    if not response:
        return fallback

    try:
        # Strip any markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", response).strip()
        data = json.loads(cleaned)
        required_keys = ["name", "tagline", "motivations", "pain_points", "recommended_actions", "narrative"]
        if all(k in data for k in required_keys):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    logger.warning(f"Could not parse Ollama JSON for segment {segment_id} — using fallback")
    return fallback


async def generate_personas(segmentation_result: dict, df: pd.DataFrame) -> list[dict]:
    """
    Generate persona profiles for each segment.
    Attempts Ollama generation with structured fallback.
    """
    available = await ollama.check_availability()
    logger.info(f"Persona generation — Ollama available: {available}")

    personas = []

    for seg_info in segmentation_result["segments"]:
        sid = seg_info["segment_id"]
        mask = df["segment"] == sid
        subset = df[mask]

        stats = {
            "label": seg_info["label"],
            "avg_ltv": seg_info["avg_ltv"],
            "avg_churn_risk": seg_info["avg_churn_risk"],
            "avg_total_spend": float(subset["total_spend"].mean()) if "total_spend" in subset else 0,
            "avg_purchase_frequency": float(subset["purchase_frequency_monthly"].mean()) if "purchase_frequency_monthly" in subset else 0,
            "avg_nps": float(subset["nps_score"].mean()) if "nps_score" in subset else 5.0,
            "avg_sessions": float(subset["session_count_30d"].mean()) if "session_count_30d" in subset else 0,
            "avg_age": float(subset["age"].mean()) if "age" in subset else 35,
            "top_income": subset["income_bracket"].mode()[0] if "income_bracket" in subset and len(subset) else "medium",
            "top_subscription": subset["subscription_type"].mode()[0] if "subscription_type" in subset and len(subset) else "basic",
        }

        fallback = FALLBACK_PERSONAS[sid % len(FALLBACK_PERSONAS)].copy()

        if available:
            prompt = _build_persona_prompt(sid, stats)
            response = await ollama.generate(prompt, system=PERSONA_SYSTEM_PROMPT)
            persona_data = _parse_ollama_response(response, sid, fallback)
        else:
            persona_data = fallback

        persona = {
            "segment_id": sid,
            "name": persona_data.get("name", fallback["name"]),
            "tagline": persona_data.get("tagline", fallback["tagline"]),
            "demographics": {
                "avg_age": round(stats["avg_age"], 0),
                "top_income": stats["top_income"],
                "top_subscription": stats["top_subscription"],
                "segment_size": seg_info["size"],
            },
            "behaviors": {
                "avg_ltv": stats["avg_ltv"],
                "avg_spend": round(stats["avg_total_spend"], 2),
                "avg_frequency": round(stats["avg_purchase_frequency"], 2),
                "avg_nps": round(stats["avg_nps"], 1),
                "churn_risk_pct": round(stats["avg_churn_risk"] * 100, 1),
            },
            "motivations": persona_data.get("motivations", fallback["motivations"]),
            "pain_points": persona_data.get("pain_points", fallback["pain_points"]),
            "recommended_actions": persona_data.get("recommended_actions", fallback["recommended_actions"]),
            "narrative": persona_data.get("narrative", fallback["narrative"]),
        }
        personas.append(persona)
        logger.info(f"Persona generated for segment {sid}: {persona['name']}")

    return personas
