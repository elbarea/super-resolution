import numpy as np
import tensorflow as tf
import pytest

from model.srgan import generator, discriminator
from train import SrganTrainer


def _tiny_srgan_dataset():
    lr = tf.constant(np.random.randint(0, 255, (1, 24, 24, 3), dtype=np.uint8))
    hr = tf.constant(np.random.randint(0, 255, (1, 96, 96, 3), dtype=np.uint8))
    return tf.data.Dataset.from_tensors((lr, hr)).repeat()


def test_srgan_trainer_runs(tmp_path):
    gen = generator()
    disc = discriminator()
    trainer = SrganTrainer(generator=gen, discriminator=disc,
                           checkpoint_dir=str(tmp_path / 'srgan'))
    ds = _tiny_srgan_dataset()
    trainer.train(ds, steps=10)
    assert trainer.checkpoint.step.numpy() == 10


def test_srgan_trainer_checkpoint_resume(tmp_path):
    ckpt_dir = str(tmp_path / 'srgan')

    gen = generator()
    disc = discriminator()
    trainer = SrganTrainer(generator=gen, discriminator=disc, checkpoint_dir=ckpt_dir)
    ds = _tiny_srgan_dataset()
    trainer.train(ds, steps=50)
    assert trainer.checkpoint.step.numpy() == 50

    # Resume: new trainer instance from same checkpoint directory
    gen2 = generator()
    disc2 = discriminator()
    trainer2 = SrganTrainer(generator=gen2, discriminator=disc2, checkpoint_dir=ckpt_dir)
    assert trainer2.checkpoint.step.numpy() == 50


def test_srgan_generator_property(tmp_path):
    gen = generator()
    disc = discriminator()
    trainer = SrganTrainer(generator=gen, discriminator=disc,
                           checkpoint_dir=str(tmp_path / 'srgan'))
    # generator and discriminator properties must return the tracked models
    assert trainer.generator is trainer.checkpoint.generator
    assert trainer.discriminator is trainer.checkpoint.discriminator
