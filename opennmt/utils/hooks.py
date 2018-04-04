"""Custom hooks."""

from __future__ import print_function

import os

import tensorflow as tf

from opennmt.utils import misc


class LogParametersCountHook(tf.train.SessionRunHook):
  """Simple hook that logs the number of trainable parameters."""

  def begin(self):
    tf.logging.info("Number of trainable parameters: %d", misc.count_parameters())


_DEFAULT_COUNTERS_COLLECTION = "counters"


def add_counter(name, tensor):
  """Registers a new counter.

  Args:
    name: The name of this counter.
    tensor: The integer ``tf.Tensor`` to count.

  See Also:
    :meth:`opennmt.utils.misc.WordCounterHook` that fetches these counters
    to log their value in TensorBoard.
  """
  count = tf.cast(tensor, tf.int64)
  total_count_init = tf.Variable(
      initial_value=0,
      name=name + "_init",
      trainable=False,
      dtype=count.dtype)
  total_count = tf.assign_add(
      total_count_init,
      count,
      name=name)
  tf.add_to_collection(_DEFAULT_COUNTERS_COLLECTION, total_count)


class CountersHook(tf.train.SessionRunHook):
  """Hook that summarizes counters.

  Implementation is mostly copied from StepCounterHook.
  """

  def __init__(self,
               every_n_steps=100,
               every_n_secs=None,
               output_dir=None,
               summary_writer=None):
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError("exactly one of every_n_steps and every_n_secs should be provided.")
    self._timer = tf.train.SecondOrStepTimer(
        every_steps=every_n_steps,
        every_secs=every_n_secs)

    self._summary_writer = summary_writer
    self._output_dir = output_dir

  def begin(self):
    self._counters = tf.get_collection(_DEFAULT_COUNTERS_COLLECTION)
    if not self._counters:
      return

    if self._summary_writer is None and self._output_dir:
      self._summary_writer = tf.summary.FileWriterCache.get(self._output_dir)

    self._last_count = [None for _ in self._counters]
    self._global_step = tf.train.get_global_step()
    if self._global_step is None:
      raise RuntimeError("Global step should be created to use WordCounterHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    if not self._counters:
      return None
    return tf.train.SessionRunArgs([self._counters, self._global_step])

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    if not self._counters:
      return

    counters, step = run_values.results
    if self._timer.should_trigger_for_step(step):
      elapsed_time, _ = self._timer.update_last_triggered_step(step)
      if elapsed_time is not None:
        for i in range(len(self._counters)):
          if self._last_count[i] is not None:
            name = self._counters[i].name.split(":")[0]
            value = (counters[i] - self._last_count[i]) / elapsed_time
            if self._summary_writer is not None:
              summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
              self._summary_writer.add_summary(summary, step)
            tf.logging.info("%s: %g", name, value)
          self._last_count[i] = counters[i]


class SaveEvaluationPredictionHook(tf.train.SessionRunHook):
  """Hook that saves the evaluation predictions."""

  def __init__(self, model, output_file, mode, post_evaluation_fn=None, best_models_dir=None):
    """Initializes this hook.

    Args:
      model: The model for which to save the evaluation predictions.
      output_file: The output filename which will be suffixed by the current
        training step.
      post_evaluation_fn: (optional) A callable that takes as argument the
        current step and the file with the saved predictions.
    """
    self._model = model
    self._output_file = output_file
    self._post_evaluation_fn = post_evaluation_fn
    self._best_external_scores = [[0, -1]]  # list initialization to record external scores
    self._best_models_dir = best_models_dir
    self._mode = mode


  def begin(self):
    self._predictions = misc.get_dict_from_collection("predictions")
    self._saver = tf.train.Saver()
    if not self._predictions:
      raise RuntimeError("The model did not define any predictions.")
    self._global_step = tf.train.get_global_step()
    if self._global_step is None:
      raise RuntimeError("Global step should be created to use SaveEvaluationPredictionHook.")
    if self._best_models_dir is not None:
        if not os.path.exists(self._best_models_dir):
            os.makedirs(self._best_models_dir)


  def before_run(self, run_context):  # pylint: disable=unused-argument
    return tf.train.SessionRunArgs([self._predictions, self._global_step])

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    predictions, self._current_step= run_values.results
    self._output_path = "{}.{}".format(self._output_file, self._current_step)
    with open(self._output_path, "a") as output_file:
      for prediction in misc.extract_batches(predictions):
        self._model.print_prediction(prediction, stream=output_file)

  def end(self, session):
    tf.logging.info("Evaluation predictions saved to %s", self._output_path)
    if self._post_evaluation_fn is not None:
      external_evaluator_scores, external_evaluator_names = self._post_evaluation_fn(self._current_step, self._output_path)
      if self._best_models_dir and external_evaluator_scores and self._mode == tf.estimator.ModeKeys.TRAIN:
        self.save_best_model(external_evaluator_scores, external_evaluator_names, session)

  def save_best_model(self, new_scores, evaluator_names, session):
    for new_score, new_evaluator_name in zip(new_scores, evaluator_names):
      if any(x[1] < new_score for x in self._best_external_scores):
        self._best_external_scores.append([self._global_step.eval(session), new_score])
        self._best_external_scores = sorted(self._best_external_scores, key=lambda k: k[1])[:10]
        save_name = "model.ckpt-{}.{}-{}".format(self._global_step.eval(session), new_evaluator_name, new_score)
        save_path = os.path.join(self._best_models_dir, save_name)
        tf.logging.info("Saving new best external evaluator model in {}".format(save_path))
        tf.logging.info(self._best_external_scores)
        self._saver.save(session, save_path)