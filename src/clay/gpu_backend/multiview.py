"""Apply multiple image conditions to one TRELLIS.2 latent trajectory."""

from contextlib import ExitStack, contextmanager
from itertools import cycle
from unittest.mock import patch


def _cycle_conditions(sampler):
    native_sample = sampler.sample
    native_step = sampler.sample_once

    def sample(model, noise, cond, *args, **kwargs):
        # Each pass starts with the front view. Keep the model's single-sample
        # shape and CFG path, including its reference-dependent rescaling.
        conditions = cycle(cond)

        def step(model, x_t, t, t_prev, _cond, **step_kwargs):
            return native_step(model, x_t, t, t_prev, next(conditions), **step_kwargs)

        with patch.object(sampler, "sample_once", step):
            return native_sample(model, noise, cond[0], *args, **kwargs)

    return sample


@contextmanager
def condition_reference_views(pipe, extra_images):
    """Cycle views in the structure, both shape passes, and material sampling.

    TRELLIS.2's unmodified run() still handles the front view and final decode.
    Encode one view at a time to retain its low-VRAM model placement behavior.
    """
    get_cond = pipe.get_cond
    extra_images = [pipe.preprocess_image(image) for image in extra_images]

    def multi_cond(images, resolution, include_neg_cond=True):
        front = get_cond(images, resolution, include_neg_cond)
        extra = (get_cond([image], resolution, False)["cond"] for image in extra_images)
        return {**front, "cond": (front["cond"], *extra)}

    with ExitStack() as stack:
        stack.enter_context(patch.object(pipe, "get_cond", multi_cond))
        for sampler in (pipe.sparse_structure_sampler, pipe.shape_slat_sampler, pipe.tex_slat_sampler):
            stack.enter_context(patch.object(sampler, "sample", _cycle_conditions(sampler)))
        yield
