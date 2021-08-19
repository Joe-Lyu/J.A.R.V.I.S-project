## Fine tuning on custom datasets

Reference:  ["Beginner’s Guide to Retrain GPT-2 (117M) to Generate Custom Text Content"](https://medium.com/@ngwaifoong92/beginners-guide-to-retrain-gpt-2-117m-to-generate-custom-text-content-8bb5363d8b7f)

To retrain GPT-2 117M model on a custom text dataset:

```
PYTHONPATH=src ./train.py --dataset <file|directory|glob>
```

If you want to precompute the dataset's encoding for multiple runs, you can instead use:

```
PYTHONPATH=src ./encode.py <file|directory|glob> /path/to/encoded.npz
PYTHONPATH=src ./train.py --dataset /path/to/encoded.npz
```

Make sure `cudnn` is installed. [Some have
reported](https://github.com/nshepperd/gpt-2/issues/8) that `train.py`
runs without it but has worse memory usage and might OOM.

### Tensor Rematerialization

Experimental: a rematerialization rewriter based on `Efficient
Rematerialization for Deep Networks`
<https://papers.nips.cc/paper/9653-efficient-rematerialization-for-deep-networks.pdf>,
which unlike gradient checkpointing works in tensorflow 2.0 and is
able to automatically select checkpoints in arbitrary graphs. Using
this I was able to finetune GPT-2 1.5B on a single graphics card using
slightly less than 12G of video ram with very little slowdown.

To use this is a little involved, because the graph optimization
algorithm is offloaded to an optimized Haskell program. First, go into
subdirectory `twremat`, and build it by invoking:

    cabal v2-install --installdir=../bin

(You'll need to install cabal if you haven't already -- but setting up
ghc and haskell compilation is beyond the scope of this README.)

Then run `train.py` as normal, enabling `--twremat` and setting
`--twremat_memlimit` to an appropriate value -- this sets the amount
of memory assumed to be available for computation of gradients, so it
should be roughly the memory size of your graphics card minus whatever
is taken up by the gpt-2 weights, and any other bookkeeping
variables. You may need to experiment with the memlimit until you find
the largest value that doesn't OOM.

(You probably also want to use SGD as optimizer instead of Adam to
minimize those bookkeeping variables, of which Adam uses a lot).

### Gradient Checkpointing

https://github.com/openai/gradient-checkpointing is included to reduce
the memory requirements of the model, and can be enabled by
`--memory_saving_gradients`. The checkpoints are currently chosen
manually (poorly) by just adding layer 10 to the 'checkpoints'
collection in model.py.

Gradient checkpointing doesn't work in tensorflow v2.0 and later due
to the removal of tf.contrib. You should use tensor rematerialization
instead if possible.

### Validation loss

Set `--val_every` to a number of steps `N > 0`, and "validation" loss
against a fixed sample of the dataset will be calculated every N steps
to get a better sense of training progress. N around 200
suggested. You can set `--val_dataset` to choose a separate validation
dataset, otherwise it defaults to a sample from the train dataset (so
not a real cross-validation loss!).

### Optimizer

You can use SGD instead of Adam with `--optimizer sgd`. This also
helps conserve memory when training larger models. Note: the learning
rate needs to be adjusted for SGD, due to not having Adam's gradient
normalization (0.0006 seems to be a good number from some
experiments).

# Original README

**Status:** Archive (code is provided as-is, no updates expected)

# gpt-2

Code and models from the paper ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

You can read about GPT-2 and its staged release in our [original blog post](https://blog.openai.com/better-language-models/), [6 month follow-up post](https://openai.com/blog/gpt-2-6-month-follow-up/), and [final post](https://www.openai.com/blog/gpt-2-1-5b-release/).

We have also [released a dataset](https://github.com/openai/gpt-2-output-dataset) for researchers to study their behaviors.

<sup>*</sup> *Note that our original parameter counts were wrong due to an error (in our previous blog posts and paper).  Thus you may have seen small referred to as 117M and medium referred to as 345M.*

## Usage

This repository is meant to be a starting point for researchers and engineers to experiment with GPT-2.

For basic information, see our [model card](./model_card.md).

### Some caveats

- GPT-2 models' robustness and worst case behaviors are not well-understood.  As with any machine-learned model, carefully evaluate GPT-2 for your use case, especially if used without fine-tuning or in safety-critical applications where reliability is important.
- The dataset our GPT-2 models were trained on contains many texts with [biases](https://twitter.com/TomerUllman/status/1101485289720242177) and factual inaccuracies, and thus GPT-2 models are likely to be biased and inaccurate as well.
- To avoid having samples mistaken as human-written, we recommend clearly labeling samples as synthetic before wide dissemination.  Our models are often incoherent or inaccurate in subtle ways, which takes more than a quick read for a human to notice.

### Work with us

Please [let us know](mailto:languagequestions@openai.com) if you’re doing interesting research with or working on applications of GPT-2!  We’re especially interested in hearing from and potentially working with those who are studying
- Potential malicious use cases and defenses against them (e.g. the detectability of synthetic text)
- The extent of problematic content (e.g. bias) being baked into the models and effective mitigations

## Development

See [DEVELOPERS.md](./DEVELOPERS.md)

## Contributors

See [CONTRIBUTORS.md](./CONTRIBUTORS.md)

## Citation

Please use the following bibtex entry:
```
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```

## Future work

We may release code for evaluating the models on various benchmarks.

We are still considering release of the larger models.

## License

[Modified MIT](./LICENSE)
