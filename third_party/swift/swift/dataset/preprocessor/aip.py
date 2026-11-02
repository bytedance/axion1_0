# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, Optional

from .core import MessagesPreprocessor, RowPreprocessor

_joint_lm_training: bool = False


def configure_joint_lm_training(enabled: bool) -> None:
    global _joint_lm_training
    _joint_lm_training = enabled


_TRUNCATED_MARKER = ' [TRUNCATED]'
_ASR_MAX_CHARS = 2048
_BG_ASR_MAX_CHARS = 2048
_OCR_MAX_CHARS = 2048
_TITLE_MAX_CHARS = None   # never truncate title/sticker
_STICKER_MAX_CHARS = None
_METADATA_MAX_CHARS = None  # no metadata-block cap; sequence-level truncation handles overflow

UNIFIED_MINOR_GUIDELINE = (
    "You are an expert video content moderator for child-safety review.\n"
    "Assess whether this video is safe and age-appropriate for users under 13. "
    "Use the video frames together with title, stickers, ASR, background ASR, OCR, and motion cues. "
    "Prioritize child safety over creator intent, and when uncertain, choose the more protective judgment.\n"
    "#### MAIN REVIEW GOAL\n"
    "Content for minors should be educational, entertaining, positive, or safely informative. "
    "Minor-unsafe content includes material that may harm a child physically, emotionally, socially, or mentally; "
    "encourage harmful imitation; normalize risky behavior; or expose children to mature themes.\n"
    "#### NOT ALLOWED / HIGH RISK\n"
    "1) Sexual or mature content: nudity or partial nudity; sexualized clothing, poses, dancing, gestures, camera framing, "
    "body-part emphasis, flirtation, sexual jokes, innuendo, explicit references, solicitation, dating/adult relationship themes, "
    "or any sexualization of youth.\n"
    "2) Eating-disorder or harmful body-image content: glorifying or normalizing disordered eating, extreme thinness, "
    "body comparison, restrictive food control, appetite suppression, purging references, unhealthy weight/muscle-change goals, "
    "or presenting suffering, control, or appearance ideals as desirable.\n"
    "3) Self-harm, suicide-related crisis, or severe mental distress: content that depicts, normalizes, glorifies, romanticizes, "
    "aestheticizes, trivializes, or gives practical guidance related to self-harm or crisis; content implying hopelessness, "
    "giving up, coded harmful coping, or that emotional pain is permanent or beyond help.\n"
    "4) Violence, danger, and fear: threats, weapons, fighting, cruelty, dangerous stunts, risky challenges, horror, disturbing imagery, "
    "bullying, coercion, peer pressure, harassment, hate, extremist content, or instructions that could cause harm.\n"
    "5) Other child-unsafe material: alcohol, tobacco, drugs, gambling, unsafe medical advice, misinformation, profanity, discrimination, "
    "requests for personal information/contact/photos/location, links/downloads/purchases, financial transactions, beauty-enhancement promotion, "
    "or adult creators soliciting direct contact, gifts, exclusive relationships, or in-person meetings with minors.\n"
    "#### AGE-RESTRICTED / NEED CAREFUL CONTEXT\n"
    "Educational or documentary discussion of puberty, sexual health, eating disorders, mental health, trauma, violence, or medical topics "
    "may be acceptable only when clearly factual, age-appropriate, non-graphic, non-sensational, and framed around safety, recovery, support, "
    "professional help, or healthy boundaries. Do not treat content as safe merely because it claims to be educational.\n"
    "#### ALLOWED / LOWER RISK\n"
    "Age-appropriate education, family-friendly entertainment, creativity, kindness, safe tutorials, supervised family or guardian interactions, "
    "basic body or health education without sexualization, body neutrality, healthy coping, emotional regulation, resilience, help-seeking, "
    "sports nutrition focused on performance/health, religious or cultural fasting, breastfeeding/caregiving, and traditional beach/pool/medical/cultural contexts when not sexualized.\n"
    "#### REVIEW NOTE\n"
    "Look for the overall framing: intent, visual emphasis, language, emotional tone, repeated motion, camera focus, and whether the content "
    "could be imitated or misunderstood by a child. Output should reflect the strongest relevant risk category even if the video also contains benign elements."
)


def _truncate_field(text: Any, max_chars) -> str:
    if text is None:
        return ''
    s = text if isinstance(text, str) else str(text)
    return s if (max_chars is None or len(s) <= max_chars) else s[:max_chars] + _TRUNCATED_MARKER


class ThemisVideoMetaPreprocessor(RowPreprocessor):
    _base_preprocessor_version: str = '5'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        suffix = 'jlm' if _joint_lm_training else ''
        self.preprocessor_version = self._base_preprocessor_version + suffix

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        frame_tos_path = row.get('frame_tos_path')
        if not frame_tos_path:
            return None

        asr = _truncate_field(row.get('asr_text', ''), _ASR_MAX_CHARS)
        bg_asr = _truncate_field(row.get('background_asr_text', ''), _BG_ASR_MAX_CHARS)
        title = _truncate_field(row.get('title', ''), _TITLE_MAX_CHARS)
        sticker = _truncate_field(row.get('sticker', ''), _STICKER_MAX_CHARS)
        ocr = _truncate_field(row.get('ocr_text', ''), _OCR_MAX_CHARS)

        metadata_parts = []
        if title and title.strip():
            metadata_parts.append(f'Title: {title}')
        if sticker and sticker.strip():
            metadata_parts.append(f'Sticker: {sticker}')
        if asr and asr.strip():
            metadata_parts.append(f'ASR text: {asr}')
        if bg_asr and bg_asr.strip():
            metadata_parts.append(f'Background ASR text: {bg_asr}')
        if ocr and ocr.strip():
            metadata_parts.append(f'OCR text: {ocr}')

        metadata_text = '\n'.join(metadata_parts)
        if _METADATA_MAX_CHARS is not None and len(metadata_text) > _METADATA_MAX_CHARS:
            metadata_text = metadata_text[:_METADATA_MAX_CHARS] + '\n[METADATA TRUNCATED]'

        prompt = f'{UNIFIED_MINOR_GUIDELINE}\n\n{metadata_text}'
        messages = [{'role': 'user', 'content': f'<video>{prompt}'}]

        ts_score = row.get('ts_score_norm')
        regression_label = None
        if ts_score is not None:
            try:
                regression_label = float(ts_score)
            except (TypeError, ValueError):
                return None

        task_type = row.get('task_type') or 'aip_regression'
        should_append_score = _joint_lm_training or task_type == 'joint_lm_training'
        if should_append_score and regression_label is not None:
            messages.append({'role': 'assistant', 'content': f'{regression_label:.4f}'})

        out: Dict[str, Any] = {'messages': messages, 'videos': [str(frame_tos_path)]}
        if regression_label is not None:
            out['regression_labels'] = regression_label
        cg_id = row.get('__cg_id')
        if cg_id is not None:
            try:
                cg_id_int = int(cg_id)
                out['__cg_id'] = cg_id_int
                out['cg_ids'] = cg_id_int
            except (TypeError, ValueError):
                pass
        total_tokens_val = row.get('total_tokens') or row.get('total_token')
        length_val = row.get('length') or total_tokens_val
        if length_val is not None:
            try:
                out['length'] = int(length_val)
            except (TypeError, ValueError):
                pass
        if total_tokens_val is not None:
            try:
                out['total_tokens'] = int(total_tokens_val)
            except (TypeError, ValueError):
                pass
        visual_tokens_val = row.get('visual_tokens') or row.get('vision_token')
        if visual_tokens_val is not None:
            try:
                out['visual_tokens'] = int(visual_tokens_val)
            except (TypeError, ValueError):
                pass
        return out


class ThemisMessagesPreprocessor(MessagesPreprocessor):
    _base_preprocessor_version: str = '3'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        suffix = 'jlm' if _joint_lm_training else ''
        self.preprocessor_version = self._base_preprocessor_version + suffix

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = super().preprocess(row)
        if row is None:
            return None
        ts_score = row.get('ts_score_norm')
        regression_label = None
        if ts_score is not None:
            try:
                regression_label = float(ts_score)
            except (TypeError, ValueError):
                return None
        if regression_label is not None:
            row['regression_labels'] = regression_label
            task_type = row.get('task_type') or 'aip_regression'
            should_append_score = _joint_lm_training or task_type == 'joint_lm_training'
            if should_append_score:
                msgs = row['messages']
                if msgs and msgs[-1]['role'] == 'assistant':
                    msgs[-1]['content'] = msgs[-1]['content'] + f'\nScore: {regression_label:.4f}'
                else:
                    msgs.append({'role': 'assistant', 'content': f'{regression_label:.4f}'})
        return row
