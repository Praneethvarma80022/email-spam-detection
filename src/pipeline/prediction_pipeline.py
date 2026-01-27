import mailbox
import pickle
import pandas as pd
from typing import Dict, List

from src.utils.state import PredictionState
from src.utils.logger import get_logger
from src.config.config import Config
from src.utils.email_utils import extract_body, clean_text

logger = get_logger(__name__)


class PredictionPipeline:

    def __init__(self, load_models: bool = True):
        self.config = Config()
        self.mailbox = None
        self.vectorizer = None
        self.model = None

        if load_models:
            self._load_models()

    def _load_models(self):
        with open(self.config.feature_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        with open(self.config.model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict_single_email(self, email_text: str) -> Dict:
        """
        Refined Sentinel Shield with CLEAN-TEXT SAFE whitelist
        """

        if self.model is None or self.vectorizer is None:
            self._load_models()

        # -------------------------------
        # A. ML PREDICTION
        # -------------------------------
        cleaned_text = clean_text(email_text)
        features = self.vectorizer.transform([cleaned_text])
        prediction = int(self.model.predict(features)[0])

        spam_conf = ham_conf = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            classes = list(self.model.classes_)
            if 1 in classes and 0 in classes:
                spam_conf = float(proba[classes.index(1)]) * 100
                ham_conf = float(proba[classes.index(0)]) * 100

        # -------------------------------
        # B. SENTINEL SHIELD (USE CLEANED TEXT ❗)
        # -------------------------------
        text_lower = cleaned_text.lower()

        # ✅ CORPORATE WHITELIST (CLEAN-TEXT FRIENDLY)
        corporate_whitelist = [
            "infosys springboard",
            "springboard support infosys",
            "meeting details",
            "attendance and individual contribution",
            "team infosys springboard"
        ]

        is_whitelisted = any(term in text_lower for term in corporate_whitelist)

        # ❌ REMOVED "internship" (major false-positive cause)
        trusted_brands = [
            "iit", "ibm", "aicte", "wipro", "cisco", "meta"
        ]

        scam_tactics = [
            "congrats", "shortlisted", "limited",
            "whatsapp", "countdown"
        ]

        brand_score = sum(1 for b in trusted_brands if b in text_lower)
        tactic_score = sum(1 for t in scam_tactics if t in text_lower)

        is_list_heavy = (
            len(cleaned_text.split()) > 200
            and "student name" in text_lower
        )

        is_shield_trigger = (
            ((brand_score >= 2 and tactic_score >= 2) or is_list_heavy)
            and not is_whitelisted
        )

        # -------------------------------
        # C. FINAL DECISION (GUARANTEED)
        # -------------------------------
        if is_whitelisted:
            label = "Ham"
            confidence = 90.0

        elif is_shield_trigger:
            label = "Spam"
            confidence = 98.5

        else:
            if prediction == 1:
                label = "Spam"
                confidence = spam_conf or 60.0
            else:
                label = "Ham"
                confidence = ham_conf or 60.0

        confidence = min(max(float(confidence), 0.0), 100.0)

        return {
            "prediction": label,
            "confidence": round(confidence, 2),
            "spam_probability": round(spam_conf, 2) if spam_conf is not None else None,
            "ham_probability": round(ham_conf, 2) if ham_conf is not None else None,
            "is_phishing_pattern": is_shield_trigger,
            "is_whitelisted": is_whitelisted
        }

    # -------------------------------
    # BATCH / MBOX PROCESSING
    # -------------------------------
    def load_mailbox(self, mailbox_path: str):
        self.mailbox = mailbox.mbox(mailbox_path)

    def process_mailbox(self, mailbox_path: str = None) -> List[Dict]:
        if mailbox_path:
            self.load_mailbox(mailbox_path)

        if self.mailbox is None:
            raise ValueError("Mailbox not loaded")

        results = []

        for msg in self.mailbox:
            body = clean_text(extract_body(msg))
            subject = clean_text(msg.get("Subject", ""))

            full_text = f"{subject} {body}"
            prediction = self.predict_single_email(full_text)

            results.append({
                "Subject": subject,
                "Body": body,
                "Prediction": prediction["prediction"],
                "Confidence": prediction["confidence"],
                "Spam Probability": prediction["spam_probability"],
                "Ham Probability": prediction["ham_probability"]
            })

        self.mailbox.close()
        return results

    def predict_mbox_file(self, mailbox_path: str, output_path: str = None) -> pd.DataFrame:
        data = self.process_mailbox(mailbox_path)
        df = pd.DataFrame(data)

        if output_path:
            df.to_csv(output_path, index=False)

        return df


def run_legacy_pipeline(state: PredictionState):
    pipeline = PredictionPipeline()
    df = pipeline.predict_mbox_file(state.mailbox_path, "data/predictions.csv")
    state.mail_data = df.to_dict(orient="records")
