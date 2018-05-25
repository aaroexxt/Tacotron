import argparse
import os
import re
import time
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


sentences = [
  # From July 8, 2017 New York Times:
  'Hey. How\'s your day going?',
  'I\'m doing pretty well, how about you?',
  'The Politics of Jane Eyre.',
  'Elizabeth, the eldest sibling of a tight family of four budding young authors, who created an epic literature for their fantasy worlds of Angria and Gondal.',
  'Scientists at the CERN laboratory say they have discovered a new particle.',
  'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
  'President Trump met with other leaders at the Group of 20 conference.',
  'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
  # From Google's Tacotron example page:
  'Generative adversarial network or variational auto-encoder.',
  'The buses aren\'t the problem, they actually provide a solution.',
  'Does the quick brown fox jump over the lazy dog?',
  'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.'
]

def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path = args.output_path
  for i, text in enumerate(sentences):
    path = '%s-%d.wav' % (base_path, i)
    print('Synthesizing: %s' % path)
    try:
      start_time = time.time()
      with open(path, 'wb') as f:
        f.write(synth.synthesize(text))
      print("Time to synthesize: "+str(round((time.time() - start_time)*100)/100)+"s"+", "+str(round(10000*(i/(len(sentences))))/100)+"% done")
    except Exception as e:
      log('Error synthesizing: %s' % e)
      traceback.print_exc()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--output_path', default='NA', help="Path to output synthesized WAV files to")
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  if (args.output_path == "NA"):
    args.output_path = get_output_base_path(args.checkpoint)
  elif (not args.output_path.endswith("/")):
    args.output_path += "/"
  print("Output path for wavs: "+args.output_path)
  run_eval(args)


if __name__ == '__main__':
  main()
