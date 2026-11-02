# Copyright (c) ModelScope Contributors. All rights reserved.
import os

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from swift.utils import get_env_args, upper_bound
from ..base import Template
from ..constant import LLMTemplateType, MLLMTemplateType
from ..register import TemplateMeta, register_template
from ..template_inputs import StdTemplateInputs
from ..utils import Context, Prompt, findall
from ..vision_utils import load_audio


@dataclass
class GemmaTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<bos>'])
    prompt: Prompt = field(
        default_factory=lambda: ['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<end_of_turn>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<end_of_turn>'])
    system_prefix: Optional[Prompt] = field(
        default_factory=lambda: ['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'])


register_template(GemmaTemplateMeta(LLMTemplateType.gemma))


class PaliGemmaTemplate(Template):
    placeholder_tokens = ['<image>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        if self.mode == 'vllm':
            self.prompt = ['{{QUERY}}']
            return []
        else:
            self.prompt = ['{{QUERY}}\n']
            return ['<image>' * self.processor.image_seq_length + '<bos>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        raw_image = inputs.images
        processor = self.processor
        if encoded['labels'] is not None:
            n = upper_bound(0, len(encoded['labels']), lambda idx: encoded['labels'][idx] == -100)
            n2 = len(encoded['labels']) - n
            encoded['token_type_ids'] = [0] * n + [1] * n2
        else:
            encoded['token_type_ids'] = [0] * len(encoded['input_ids'])
        if raw_image:
            model_inputs = processor(text='<image>' * len(raw_image), images=raw_image, return_tensors='pt')
            encoded['pixel_values'] = model_inputs['pixel_values'].to(self.model_info.torch_dtype)
        return encoded


register_template(
    TemplateMeta(
        MLLMTemplateType.paligemma,
        prefix=[],
        prompt=['{{QUERY}}\n'],
        chat_sep=None,
        suffix=['<eos>'],
        template_cls=PaliGemmaTemplate,
    ))


@dataclass
class Gemma3TextTemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<bos>'])
    prompt: Prompt = field(
        default_factory=lambda: ['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<end_of_turn>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<end_of_turn>'])


class Gemma3Template(Template):

    def _swift_encode(self, inputs: StdTemplateInputs):
        if inputs.system is not None:
            system = inputs.system
            inputs.system = None
            inputs.messages[0]['content'] = system + '\n\n' + inputs.messages[0]['content']
        for message in inputs.messages:
            if message['role'] == 'assistant' and isinstance(message['content'], str):
                message['content'] = message['content'].strip('\n')
        return super()._swift_encode(inputs)


register_template(Gemma3TextTemplateMeta(LLMTemplateType.gemma3_text, template_cls=Gemma3Template))


class Gemma3VisionTemplate(Gemma3Template):
    boi_token_id = 255999
    placeholder_tokens = ['<start_of_image>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        assert media_type == 'image'
        return ['<start_of_image>']

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from transformers.models.gemma3.processing_gemma3 import Gemma3ProcessorKwargs

        encoded = super()._encode(inputs)
        if inputs.images:
            input_ids = encoded['input_ids']
            labels = encoded['labels']
            loss_scale = encoded.get('loss_scale', None)
            idx_list = findall(input_ids, self.boi_token_id)
            img_tokens = self._tokenize(self.processor.full_image_sequence)
            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                lambda _: img_tokens)

            # TODO: customize
            processor_kwargs = Gemma3ProcessorKwargs._defaults['images_kwargs']
            image_inputs = self.processor.image_processor(inputs.images, **processor_kwargs)
            image_inputs['pixel_values'] = torch.as_tensor(np.array(image_inputs['pixel_values']))
            image_inputs.pop('num_crops')

            array_ids = np.array(input_ids)
            mm_token_type_ids = np.zeros_like(input_ids)
            mm_token_type_ids[array_ids == self.processor.image_token_id] = 1
            encoded['token_type_ids'] = mm_token_type_ids.tolist()
            encoded['input_ids'] = input_ids
            encoded['pixel_values'] = image_inputs['pixel_values']
            encoded['labels'] = labels
            encoded['loss_scale'] = loss_scale
        return encoded


register_template(GemmaTemplateMeta(MLLMTemplateType.gemma3_vision, template_cls=Gemma3VisionTemplate))


class Gemma3nTemplate(Gemma3Template):
    boi_token_id = 255999
    boa_token_id = 256000
    placeholder_tokens = ['<start_of_image>', '<start_of_audio>']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            if self.mode == 'vllm':
                return ['<image_soft_token>']
            else:
                return ['\n\n<start_of_image>']
        elif media_type == 'audio':
            if self.mode == 'vllm':
                raise ValueError('Audio is not supported in vLLM')
            inputs.audios[index] = load_audio(inputs.audios[index], self.processor.feature_extractor.sampling_rate)
            return ['<start_of_audio>']
        else:
            raise ValueError(f'Unsupported media type: {media_type}. Supported types are: image, audio')

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        from transformers.models.gemma3n.processing_gemma3n import Gemma3nProcessorKwargs

        # Input validation
        if not inputs.images and not inputs.audios and not inputs.messages:
            raise ValueError('Provide at least one of `images`, `audios`, or `messages`.')

        encoded = super()._encode(inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)

        # Initialize token_type_ids and other outputs
        array_ids = np.array(input_ids)
        mm_token_type_ids = np.zeros_like(input_ids)

        # Handle images
        if inputs.images:
            idx_list = findall(input_ids, self.boi_token_id)
            img_tokens = self._tokenize(processor.full_image_sequence[2:])
            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                lambda _: img_tokens)

            # Process images
            processor_kwargs = Gemma3nProcessorKwargs._defaults.get('images_kwargs', {})
            image_inputs = processor.image_processor(inputs.images, **processor_kwargs)
            image_inputs['pixel_values'] = torch.as_tensor(
                np.array(image_inputs['pixel_values']), dtype=self.model_info.torch_dtype)
            if 'num_crops' in image_inputs:
                image_inputs.pop('num_crops')
            encoded.update(image_inputs)

        # Handle audios
        if inputs.audios:
            audio_idx_list = findall(input_ids, self.boa_token_id)
            if audio_idx_list:
                # Get audio token sequence from processor
                audio_tokens = self._tokenize(processor.full_audio_sequence)
                input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, audio_idx_list,
                                                                    lambda _: audio_tokens)

                # Process audios
                processor_kwargs = Gemma3nProcessorKwargs._defaults.get('audio_kwargs', {})
                audio_inputs = processor.feature_extractor(inputs.audios, **processor_kwargs)

                if 'input_features' in audio_inputs:
                    audio_inputs['input_features'] = torch.tensor(audio_inputs['input_features']).to(
                        self.model_info.torch_dtype)
                if 'input_features_mask' in audio_inputs:
                    audio_inputs['input_features_mask'] = torch.tensor(audio_inputs['input_features_mask'])
                encoded.update(audio_inputs)

        # Update array_ids after token extension
        array_ids = np.array(input_ids)
        mm_token_type_ids = np.zeros_like(input_ids)

        if hasattr(processor, 'image_token_id') and processor.image_token_id is not None:
            mm_token_type_ids[array_ids == processor.image_token_id] = 1

        if hasattr(processor, 'audio_token_id') and processor.audio_token_id is not None:
            mm_token_type_ids[array_ids == processor.audio_token_id] = 3

        encoded['token_type_ids'] = mm_token_type_ids.tolist()
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle multimodal data collation for Gemma3n, including audio features"""
        res = super()._data_collator_mm_data(batch)

        # Handle audio features like other templates do
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        input_features_mask = [b['input_features_mask'] for b in batch if b.get('input_features_mask') is not None]

        if input_features:
            res['input_features'] = torch.concat(input_features)
        if input_features_mask:
            res['input_features_mask'] = torch.concat(input_features_mask)

        return res


register_template(GemmaTemplateMeta(MLLMTemplateType.gemma3n, template_cls=Gemma3nTemplate))


class Gemma4Template(Template):
    placeholder_tokens = ['<|image|>', '<|audio|>', '<|video|>']
    # Video frames are laid out as `{timestamp} <|image>{<|video|> pads}<image|>` per frame
    # (processing_gemma4.replace_video_token; boi/eoi are shared with images). Truncation must
    # keep these boi/eoi tokens: get_block_sequence_ids_for_mask derives the per-frame
    # bidirectional attention blocks from CONTIGUOUS mm-token runs, so deleting the separators
    # silently merges adjacent frames into one block (no crash — the masked_scatter count still
    # matches — but a train/inference distribution shift). See Template._truncate.
    truncation_structure_tokens = ['<|image>', '<image|>', '<|audio>', '<audio|>']
    support_padding_free = True

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == 'image':
            return ['<|image|>']
        elif media_type == 'audio':
            if self.mode != 'vllm':
                inputs.audios[index] = load_audio(inputs.audios[index], self.processor.feature_extractor.sampling_rate)
            return ['<|audio|>']
        elif media_type == 'video':
            video_path = inputs.videos[index]
            is_tos = (isinstance(video_path, str) and video_path.endswith('.zip') and not os.path.isabs(video_path)
                      and not video_path.startswith(('http://', 'https://', 'data:')))
            if self.mode == 'vllm':
                if is_tos:
                    # AIP TOS frame-zip: decode to a pre-sampled clip + minimal metadata so the
                    # gemma4 vLLM video processor doesn't try to open it as a video file.
                    arr = self._load_tos_video_array(video_path)
                    n = arr.shape[0]
                    fps = float(get_env_args('video_fps', float, 2.0))
                    meta = {
                        'total_num_frames': n, 'fps': fps, 'duration': n / fps if fps else 0.0,
                        'video_backend': 'tos_zip', 'frames_indices': list(range(n)), 'do_sample_frames': False,
                    }
                    inputs.videos[index] = [(arr, meta)]
                else:
                    from vllm.assets.video import video_get_metadata, video_to_ndarrays
                    num_frames = self.processor.video_processor.num_frames
                    inputs.videos[index] = [(video_to_ndarrays(video_path, num_frames),
                                             video_get_metadata(video_path, num_frames))]
            elif is_tos:
                # Training path: feed the gemma4 video processor a pre-sampled [T,H,W,C] clip
                # (do_sample_frames is disabled in _encode for these; it resizes each frame to
                # the GEMMA4_VIDEO_MAX_SOFT_TOKENS budget).
                inputs.videos[index] = self._load_tos_video_array(video_path)
            return ['<|video|>']

    def _load_tos_video_array(self, video_path: str) -> np.ndarray:
        """Decode a TOS frame-zip to a [T, H, W, C] uint8 array (frames normalized to the
        first frame's size so they stack; the gemma4 processor does the final resize)."""
        from ..vision_utils import load_video_tos_zip
        num_frames = get_env_args('gemma4_video_num_frames', int, self.processor.video_processor.num_frames)
        frames = load_video_tos_zip(video_path, max_frames=num_frames)
        ref = frames[0].convert('RGB')
        arrs = [np.asarray(ref)]
        for f in frames[1:]:
            f = f.convert('RGB')
            if f.size != ref.size:
                f = f.resize(ref.size)
            arrs.append(np.asarray(f))
        return np.stack(arrs)

    def _get_system(self, inputs: StdTemplateInputs) -> Optional[str]:
        system = super()._get_system(inputs)
        if self._get_enable_thinking(inputs):
            system = '<|think|>\n' + (system or '')
        return system

    def _add_non_thinking_prefix(self, inputs: StdTemplateInputs, thinking_prefix: str = '<|channel>thought'):
        return super()._add_non_thinking_prefix(inputs, thinking_prefix=thinking_prefix)

    def _remove_thinking_content(self, content: str, thinking_suffix: str = '<channel|>') -> str:
        return super()._remove_thinking_content(content, thinking_suffix=thinking_suffix)

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        split_token = self._tokenize('\n')
        # Per-frame visual token budget for video (gemma4 uses video_processor.max_soft_tokens
        # ∈ {70,140,280,560,1120} per frame + num_frames, not qwen's VIDEO_MAX_TOKEN_NUM).
        video_kwargs: Dict[str, Any] = {}
        if inputs.videos:
            # Frames pre-decoded by replace_tag (TOS path) are [T,H,W,C] arrays — don't resample.
            if any(isinstance(v, np.ndarray) for v in inputs.videos):
                video_kwargs['do_sample_frames'] = False
            max_soft_tokens = get_env_args('gemma4_video_max_soft_tokens', int, 0)
            if max_soft_tokens:
                video_kwargs['max_soft_tokens'] = max_soft_tokens
            num_frames = get_env_args('gemma4_video_num_frames', int, 0)
            if num_frames:
                video_kwargs['num_frames'] = num_frames
        media_inputs = self.processor(
            text='\n'.join(['<|image|>'] * len(inputs.images) + ['<|video|>'] * len(inputs.videos)
                           + ['<|audio|>'] * len(inputs.audios)),
            audio=inputs.audios or None,
            images=inputs.images or None,
            videos=inputs.videos or None,
            return_tensors='pt',
            add_special_tokens=False,
            **video_kwargs,
        )
        splited_tokens = self._split_list(media_inputs['input_ids'][0].tolist(), split_token)
        media_inputs.pop('input_ids')
        media_inputs.pop('attention_mask')
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        mm_mask = [False] * len(input_ids)

        idx_list = []
        for key in ['image', 'video', 'audio']:
            idx_list += findall(input_ids, getattr(self.config, f'{key}_token_id'))
        sorted_order = sorted(range(len(idx_list)), key=lambda i: idx_list[i])
        idx_list = [idx_list[i] for i in sorted_order]
        splited_tokens = [splited_tokens[i] for i in sorted_order]

        def _get_new_tokens(i):
            return splited_tokens[i]

        if idx_list:
            input_ids, labels, loss_scale, mm_mask = self._extend_tokens(
                input_ids, labels, loss_scale, idx_list, _get_new_tokens, mm_mask=mm_mask)
        for key in [
                'pixel_values', 'image_position_ids', 'pixel_values_videos', 'video_position_ids', 'input_features',
                'input_features_mask'
        ]:
            if key in media_inputs:
                encoded[key] = media_inputs[key]
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        encoded['mm_token_type_ids'] = self.create_mm_token_type_ids(input_ids, mm_mask)
        return encoded

    @staticmethod
    def _cat_patch_padded(tensors: List[torch.Tensor], pad_value: float) -> torch.Tensor:
        """Concat (..., n_patches, x) tensors along dim 0 after right-padding the patch dim
        (dim -2) to the batch-wide max. gemma4 pads each sample's patches to a per-sample
        ``max_patches`` (extra patches marked (-1,-1) in *_position_ids and masked by the vision
        tower), so packed samples can disagree on that dim; pad to the common max before concat."""
        max_p = max(t.shape[-2] for t in tensors)
        out = []
        for t in tensors:
            if t.shape[-2] < max_p:
                pad = t.new_full((*t.shape[:-2], max_p - t.shape[-2], t.shape[-1]), pad_value)
                t = torch.cat([t, pad], dim=-2)
            out.append(t)
        return torch.concat(out, dim=0)

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # gemma4 media tensors carry a per-sample ``max_patches`` dim (and video adds a
        # (num_videos, num_frames) prefix), so the base concat-on-dim0 fails when packed samples
        # differ. Merge them here with patch-dim padding instead of calling super().
        res = {}
        # images: pixel_values (num_images, max_patches, patch_pixels) + image_position_ids (..., 2)
        pixel_values = [b['pixel_values'] for b in batch if b.get('pixel_values') is not None]
        if pixel_values:
            res['pixel_values'] = self._cat_patch_padded(pixel_values, 0)
            ipos = [b['image_position_ids'] for b in batch if b.get('image_position_ids') is not None]
            if ipos:
                res['image_position_ids'] = self._cat_patch_padded(ipos, -1)
        # videos: pixel_values_videos (num_videos, num_frames, max_patches, patch_pixels). The model's
        # get_video_features flattens (num_videos, num_frames) into a per-frame axis, so flatten each
        # sample's frames, pad patches, concat, and restore a singleton video axis. This preserves
        # every frame (and thus the exact per-sample soft-token count) while letting num_frames and
        # max_patches differ across packed samples.
        pvv = [b['pixel_values_videos'] for b in batch if b.get('pixel_values_videos') is not None]
        if pvv:
            res['pixel_values_videos'] = self._cat_patch_padded([v.flatten(0, 1) for v in pvv], 0).unsqueeze(1)
            vpos = [b['video_position_ids'] for b in batch if b.get('video_position_ids') is not None]
            if vpos:
                res['video_position_ids'] = self._cat_patch_padded([p.flatten(0, 1) for p in vpos], -1).unsqueeze(1)
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        if input_features:
            input_features_mask = [b['input_features_mask'] for b in batch if b.get('input_features_mask') is not None]
            max_len = max([x.shape[1] for x in input_features_mask])
            res['input_features'] = torch.concat([F.pad(x, (0, 0, 0, max_len - x.shape[1])) for x in input_features])
            res['input_features_mask'] = torch.concat(
                [F.pad(x, (0, max_len - x.shape[1])) for x in input_features_mask])
        return res


@dataclass
class Gemma4TemplateMeta(TemplateMeta):
    prefix: Prompt = field(default_factory=lambda: ['<bos>'])
    prompt: Prompt = field(default_factory=lambda: ['<|turn>user\n{{QUERY}}<turn|>\n<|turn>model\n'])
    chat_sep: Optional[Prompt] = field(default_factory=lambda: ['<turn|>\n'])
    suffix: Prompt = field(default_factory=lambda: ['<turn|>\n'])
    system_prefix: Optional[Prompt] = field(default_factory=lambda: ['<bos><|turn>system\n{{SYSTEM}}<turn|>\n'])


register_template(Gemma4TemplateMeta(MLLMTemplateType.gemma4_nothinking, template_cls=Gemma4Template))

register_template(
    Gemma4TemplateMeta(
        MLLMTemplateType.gemma4,
        template_cls=Gemma4Template,
        is_thinking=True,
        non_thinking_prefix='<|channel>thought\n<channel|>'))
