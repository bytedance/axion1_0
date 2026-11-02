# Copyright (c) ModelScope Contributors. All rights reserved.
from .bt_sft import BTSFTLoss
from .causal_lm import CustomCrossEntropyLoss
from .embedding import ContrastiveLoss, CosineSimilarityLoss, InfonceLoss, OnlineContrastiveLoss
from .reranker import ListwiseRerankerLoss, PointwiseRerankerLoss

loss_map = {
    'cross_entropy': CustomCrossEntropyLoss,  # examples
    # multi-CG BT-SFT (verbalized Bradley-Terry scoring) on the transformers backend
    'bt_sft': BTSFTLoss,
    # embedding
    'cosine_similarity': CosineSimilarityLoss,
    'contrastive': ContrastiveLoss,
    'online_contrastive': OnlineContrastiveLoss,
    'infonce': InfonceLoss,
    # # reranker
    'pointwise_reranker': PointwiseRerankerLoss,
    'listwise_reranker': ListwiseRerankerLoss,
}
