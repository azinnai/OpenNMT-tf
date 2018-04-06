"""Vocabulary utilities for Python scripts."""

import six

import tensorflow as tf


class Vocab(object):
  """Vocabulary class."""

  def __init__(self, special_tokens=None):
    """Initializes a vocabulary.

    Args:
      special_tokens: A list of special tokens (e.g. start of sentence).
    """
    self._token_to_id = {}
    self._id_to_token = []
    self._frequency = []
    self._special_tokens = special_tokens

    if self._special_tokens is not None:
      for index, token in enumerate(self._special_tokens):
        self._token_to_id[token] = index
        self._id_to_token.insert(index, token)

        # Set a very high frequency to avoid special tokens to be pruned. Note that Python sort
        # functions are stable which means that special tokens in pruned vocabularies will have
        # the same index.
        self._frequency.insert(index, float("inf"))

  @property
  def size(self):
    """Returns the number of entries of the vocabulary."""
    return len(self._id_to_token)

  def add_from_text(self, filename, tokenizer=None):
    """Fills the vocabulary from a text file.

    Args:
      filename: The file to load from.
      tokenizer: A callable to tokenize a line of text.
    """
    with open(filename, "rb") as text:
      for line in text:
        line = tf.compat.as_text(line.strip())
        if tokenizer:
          tokens = tokenizer.tokenize(line)
        else:
          tokens = line.split()
        for token in tokens:
          self.add(token)

  def serialize(self, path):
    """Writes the vocabulary on disk.

    Args:
      path: The path where the vocabulary will be saved.
    """
    with open(path, "wb") as vocab:
      for token in self._id_to_token:
        vocab.write(tf.compat.as_bytes(token))
        vocab.write(b"\n")

  def add(self, token):
    """Adds a token or increases its frequency.

    Args:
      token: The string to add.
    """
    if token not in self._token_to_id:
      index = self.size
      self._token_to_id[token] = index
      self._id_to_token.append(token)
      self._frequency.append(1)
    else:
      self._frequency[self._token_to_id[token]] += 1

  def lookup(self, identifier, default=None):
    """Lookups in the vocabulary.

    Args:
      identifier: A string or an index to lookup.
      default: The value to return if :obj:`identifier` is not found.

    Returns:
      The value associated with :obj:`identifier` or :obj:`default`.
    """
    value = None

    if isinstance(identifier, six.string_types):
      if identifier in self._token_to_id:
        value = self._token_to_id[identifier]
    elif identifier < self.size:
      value = self._id_to_token[identifier]

    if value is None:
      return default
    else:
      return value

  def prune(self, max_size=0, min_frequency=1):
    """Creates a pruned version of the vocabulary.

    Args:
      max_size: The maximum vocabulary size.
      min_frequency: The minimum frequency of each entry.

    Returns:
      A new vocabulary.
    """
    sorted_ids = sorted(range(self.size), key=lambda k: self._frequency[k], reverse=True)
    new_size = len(sorted_ids)

    # Discard words that do not meet frequency requirements.
    for i in range(new_size - 1, 0, -1):
      index = sorted_ids[i]
      if self._frequency[index] < min_frequency:
        new_size -= 1
      else:
        break

    # Limit absolute size.
    if max_size > 0:
      new_size = min(new_size, max_size)

    new_vocab = Vocab()

    for i in range(new_size):
      index = sorted_ids[i]
      token = self._id_to_token[index]
      frequency = self._frequency[index]

      new_vocab._token_to_id[token] = i  # pylint: disable=protected-access
      new_vocab._id_to_token.append(token)  # pylint: disable=protected-access
      new_vocab._frequency.append(frequency)  # pylint: disable=protected-access

    return new_vocab

  def prune_embeddings(self, embeddings_path, embedding_size):
    new_vocab = Vocab(self._special_tokens)
    embeddings_keys = self._load_embeddings_keys(embeddings_path)

    idx = 0
    for i, token in enumerate(self._id_to_token):
      if token in embeddings_keys:
        frequency = self._frequency[i]

        new_vocab._token_to_id[token] = idx  # pylint: disable=protected-access
        new_vocab._id_to_token.append(token)  # pylint: disable=protected-access
        new_vocab._frequency.append(frequency)  # pylint: disable=protected-access

    pruned_embeddings_keys = [x for x in new_vocab._token_to_id.keys()]
    embeddings = self._load_embeddings(embeddings_path, pruned_embeddings_keys)

    embeddings_path = embeddings_path.split('.')
    path, lang = ".".join(embeddings_path[:-1]), embeddings_path[-1]
    save_path = path+'.pruned.'+lang
    self._serialize_embeddings(embeddings, save_path, embedding_size)
    return new_vocab

  def _serialize_embeddings(self, embeddings, output_path, embedding_size):
    with open(output_path, 'w') as output_file:
      for token, embedding in embeddings.items():
        if len(embedding) == embedding_size: # check for bad length embeddings
          output_file.write("{} {}\n".format(token, " ".join(embedding)))

  @staticmethod
  def _load_embeddings(embeddings_path, embeddings_keys):
    embeddings = {}
    with open(embeddings_path, 'r') as embeddings_file:
      header = next(embeddings_file).strip().split()
      first_line = next(embeddings_file).strip().split()

      if len(header) == len(first_line):
        key = header[0]
        embedding = header[1:]
        if key in embeddings_keys:
          embeddings[key] = embedding

      key = first_line[0]
      embedding = first_line[1:]
      if key in embeddings_keys:
        embeddings[key] = embedding

      for line in embeddings_file:
        line = line.strip().split()
        key = line[0]
        embedding = line[1:]
        if key in embeddings_keys:
          embeddings[key] = embedding

    return embeddings

  @staticmethod
  def _load_embeddings_keys(embeddings_path):
    embeddings_keys = set()
    with open(embeddings_path, 'r') as embeddings_file:
      header = next(embeddings_file).strip().split()
      first_line = next(embeddings_file).strip().split()

      if len(header) == len(first_line):
        embeddings_keys.add(header[0])
      embeddings_keys.add(first_line[0])

      for line in embeddings_file:
        line = line.split()
        token = line[0]
        embeddings_keys.add(token)
    return embeddings_keys
