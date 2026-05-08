import numpy as np
import tensorflow as tf
import pytest

from model.edsr import edsr
from train import EdsrTrainer, WdsrTrainer, SrganGeneratorTrainer


def _tiny_dataset(scale=2):
    lr = tf.constant(np.random.randint(0, 255, (1, 8, 8, 3), dtype=np.uint8))
    hr = tf.constant(np.random.randint(0, 255, (1, 8 * scale, 8 * scale, 3), dtype=np.uint8))
    return tf.data.Dataset.from_tensors((lr, hr)).repeat()


def test_edsr_trainer_runs_one_step(tmp_path):
    model = edsr(scale=2, num_res_blocks=2, num_filters=8)
    trainer = EdsrTrainer(model=model, checkpoint_dir=str(tmp_path / 'ckpt'))
    ds = _tiny_dataset(scale=2)
    trainer.train(ds, ds.take(1), steps=1, evaluate_every=1, save_best_only=False)


def test_edsr_trainer_checkpoint_resume(tmp_path):
    ckpt_dir = str(tmp_path / 'ckpt')
    model = edsr(scale=2, num_res_blocks=2, num_filters=8)
    trainer = EdsrTrainer(model=model, checkpoint_dir=ckpt_dir)
    ds = _tiny_dataset(scale=2)
    trainer.train(ds, ds.take(1), steps=3, evaluate_every=1, save_best_only=False)
    assert trainer.checkpoint.step.numpy() == 3

    # Resume: new trainer instance, same checkpoint directory
    model2 = edsr(scale=2, num_res_blocks=2, num_filters=8)
    trainer2 = EdsrTrainer(model=model2, checkpoint_dir=ckpt_dir)
    assert trainer2.checkpoint.step.numpy() == 3


def test_edsr_trainer_step_count_increments(tmp_path):
    model = edsr(scale=2, num_res_blocks=2, num_filters=8)
    trainer = EdsrTrainer(model=model, checkpoint_dir=str(tmp_path / 'ckpt'))
    ds = _tiny_dataset(scale=2)
    trainer.train(ds, ds.take(1), steps=5, evaluate_every=10, save_best_only=False)
    assert trainer.checkpoint.step.numpy() == 5
