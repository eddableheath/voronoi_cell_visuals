"""
Load the depth estimation model
"""

import transformers


def load_model_and_fe(model: str):
    """
    Load the model and feature extractor
    """
    return (
        transformers.DPTForDepthEstimation.from_pretrained(
            model, low_cpu_mem_usage=True
        ),
        transformers.DPTImageProcessor.from_pretrained(model),
    )
