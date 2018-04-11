"""Example of a translation client."""

from __future__ import print_function

import argparse
import subprocess
import os

import tensorflow as tf

from sys import stdout

from grpc.beta import implementations

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

class TranslatorClient(object):

    def __init__(self,
                 model_name,
                 host,
                 port,
                 timeout,
                 src_lang,
                 trg_lang,
                 preprocess_script,
                 postprocess_script):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.timeout = timeout
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.postprocess_script = postprocess_script
        self.preprocess_script = preprocess_script
        self.channel = implementations.insecure_channel(host, port)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)

    @staticmethod
    def parse_translation_result(result):
      """Parses a translation result.

      Args:
        result: A `PredictResponse` proto.

      Returns:
        A list of tokens.
      """
      lengths = tf.make_ndarray(result.outputs["length"])[0]
      hypotheses = tf.make_ndarray(result.outputs["tokens"])[0]

      # Only consider the first hypothesis (the best one).
      best_hypothesis = hypotheses[0]
      best_length = lengths[0]

      return best_hypothesis[0:best_length - 1] # Ignore </s>

    def translate(self, tokens):
      """Translates a sequence of tokens.

      Args:
        stub: The prediction service stub.
        model_name: The model to request.
        tokens: A list of tokens.
        timeout: Timeout after this many seconds.

      Returns:
        A future.
      """
      length = len(tokens)

      request = predict_pb2.PredictRequest()
      request.model_spec.name = self.model_name
      request.inputs["tokens"].CopyFrom(
          tf.make_tensor_proto([tokens], shape=(1, length)))
      request.inputs["length"].CopyFrom(
          tf.make_tensor_proto([length], shape=(1,)))

      return self.stub.Predict.future(request, self.timeout)

    def preprocess(self, data):
        FNULL = open(os.devnull, 'w')
        current_dir = os.getcwd()
        process = " ".join([os.path.join(current_dir, self.preprocess_script), self.src_lang])
        p = subprocess.Popen(process,
                              shell=True,
                              stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE,
                              stderr=FNULL)
        preprocessed_data = p.communicate(data.encode('utf-8'))[0].decode('utf-8').strip()
        preprocessed_data = [line.split() for line in preprocessed_data.split('\n')]
        return preprocessed_data

    def postprocess(self, data, lang=None):
        if lang is None:
            lang = self.trg_lang
        FNULL = open(os.devnull, 'w')
        current_dir = os.getcwd()
        process = " ".join([os.path.join(current_dir, self.postprocess_script), lang])
        if isinstance(data[0][0], bytes):
            data = "\n".join([b" ".join(line).decode() for line in data])
        else:
            data = "\n".join([" ".join(line) for line in data])
        p = subprocess.Popen(process,
                              shell=True,
                              stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE,
                              stderr=FNULL)
        postprocessed_data = p.communicate(data.encode('utf-8'))[0].decode('utf-8').strip()
        postprocessed_data = postprocessed_data.split('\n')
        return postprocessed_data

    def translate_file(self, input_file, output=stdout):

        if output is not stdout:
            output = open(output, 'a')

        with open(input_file, 'r') as f:
            full_tokens = self.preprocess(f.read())

        step = 0
        batch_size = 1

        while step * batch_size < len(full_tokens):
            batch_tokens = full_tokens[step * batch_size:(step + 1) * batch_size]

            step += 1
            futures = []

            for tokens in batch_tokens:
                future = self.translate(tokens)
                futures.append(future)
            results = [self.parse_translation_result(future.result()) for future in futures]
            results = self.postprocess(results)
            batch_tokens = self.postprocess(batch_tokens, self.src_lang)

            for tokens, result in zip(batch_tokens, results):
                print("{}\n{}\n".format(tokens, result), file=output)

    def translate_from_cmd(self):
        try:
            print("Type the sentence you want to translate")
            while True:
                input_ = input()
                input_preprocessed = self.preprocess(input_)[0]
                translation = self.translate(input_preprocessed)
                translation = [self.parse_translation_result(translation.result())]
                translation = self.postprocess(translation)[0]
                print(translation)
        except KeyboardInterrupt:
            return

def main():
  parser = argparse.ArgumentParser(description="Translation client example")
  parser.add_argument("--model_name", required=True,
                      help="model name")
  parser.add_argument("--host", default="localhost",
                      help="model server host")
  parser.add_argument("--port", type=int, default=9000,
                      help="model server port")
  parser.add_argument("--timeout", type=float, default=100000.0,
                      help="request timeout")
  parser.add_argument("--src_lang", required=True,
                      help="source language")
  parser.add_argument("--trg_lang", required=True,
                      help="target language")
  parser.add_argument("--preprocess_script", required=True,
                      help="The absolute path to preprocess script")
  parser.add_argument("--postprocess_script", required=True,
                      help="The absolute path to postprocess script")
  parser.add_argument("--input_file",
                      help="The file with the sentences to be translated, one per row."
                           "If not provided, you can type a sentence from the command line.")
  parser.add_argument("--output_file",
                      help="The file were to store the translations."
                           "If not provided, the translations will be printed to STDOUT.")

  args = parser.parse_args()

  translator_client = TranslatorClient(args.model_name,
                                       args.host,
                                       args.port,
                                       args.timeout,
                                       args.src_lang,
                                       args.trg_lang,
                                       args.preprocess_script,
                                       args.postprocess_script)

  if args.input_file:
      translator_client.translate_file(args.input_file)
  else:
      translator_client.translate_from_cmd()



if __name__ == "__main__":
  main()
